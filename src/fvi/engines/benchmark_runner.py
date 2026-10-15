"""
Golden Triad Benchmark Execution Engines:
  1. Centralized Upper Bound (`run_centralized_upper_bound`)
  2. Isolated Local Silos Lower Bound (`run_isolated_local_silos`)
  3. Iterative Federated Learning (`run_iterative_fl`: FedAvg, FedProx, FedAdam)
  4. Two-Phase One-Shot Ensembling (`run_fens_fl`: FENS with INT8 Quantization)

Uses PyTorch + CUDA + BF16 automatically when `torch` is installed (e.g., on
your local Ubuntu machine and on `rama` L40S), and includes a built-in pure-NumPy
neural network trainer fallback so full end-to-end benchmarks can also run in
lightweight environments without PyTorch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fvi.config import BenchmarkConfig, HardwareConfig, ModelConfig
from fvi.ingestion.loader import LoadedDatasetBundle
from fvi.metrics.evaluator import ClassificationMetrics, compute_classification_metrics
from fvi.models.factory import TORCH_AVAILABLE, ModelSpec, build_unified_model_spec
from fvi.partition.splitter import PartitionBundle

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from fvi.models.factory import build_torch_client_model


@dataclass
class AlgorithmRunResult:
    """Stores accuracy, Macro-F1, convergence history, and communication cost for a run."""

    name: str
    category: str  # "centralized", "isolated", "iterative_fl", "one_shot_fl"
    test_metrics: ClassificationMetrics
    val_accuracy: float
    best_round: int
    total_rounds: int
    comm_bytes: int
    wall_time_sec: float
    round_history: List[Dict[str, float]] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def comm_mib(self) -> float:
        return self.comm_bytes / (1024.0 ** 2)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "test_accuracy": round(self.test_metrics.accuracy, 2),
            "test_macro_f1": round(self.test_metrics.macro_f1, 2),
            "test_weighted_f1": round(self.test_metrics.weighted_f1, 2),
            "per_class_accuracy": [
                round(v, 2) for v in self.test_metrics.per_class_accuracy
            ],
            "confusion_matrix": self.test_metrics.confusion_matrix,
            "val_accuracy": round(self.val_accuracy, 2),
            "best_round": self.best_round,
            "total_rounds": self.total_rounds,
            "comm_mib": round(self.comm_mib, 3),
            "wall_time_sec": round(self.wall_time_sec, 2),
            "round_history": self.round_history,
            "extra_info": self.extra_info,
        }


@dataclass
class IsolatedSilosResult:
    """Stores per-client and sample-weighted aggregate metrics for Isolated Local Silos."""

    weighted_test_accuracy: float
    weighted_test_macro_f1: float
    unweighted_mean_accuracy: float
    min_client_accuracy: float
    max_client_accuracy: float
    std_client_accuracy: float
    per_client_accuracy: List[float]
    per_client_macro_f1: List[float]
    per_client_samples: List[int]
    wall_time_sec: float

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "weighted_test_accuracy": round(self.weighted_test_accuracy, 2),
            "weighted_test_macro_f1": round(self.weighted_test_macro_f1, 2),
            "unweighted_mean_accuracy": round(self.unweighted_mean_accuracy, 2),
            "min_client_accuracy": round(self.min_client_accuracy, 2),
            "max_client_accuracy": round(self.max_client_accuracy, 2),
            "std_client_accuracy": round(self.std_client_accuracy, 2),
            "per_client_accuracy": [round(a, 2) for a in self.per_client_accuracy],
            "per_client_macro_f1": [round(f, 2) for f in self.per_client_macro_f1],
            "per_client_samples": self.per_client_samples,
            "wall_time_sec": round(self.wall_time_sec, 2),
        }


# ============================================================================
# Pure-NumPy Neural Network Engine (Used when PyTorch is not installed)
# ============================================================================


def _np_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _np_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class _NumpyMLP:
    """2-Hidden-Layer ReLU MLP in pure NumPy supporting FedAvg, FedProx, FedAdam, and INT8 quantization."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, seed: int = 42
    ) -> None:
        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32),
            "b1": np.zeros((1, hidden_dim), dtype=np.float32),
            "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes)).astype(np.float32),
            "b2": np.zeros((1, num_classes), dtype=np.float32),
        }

    def clone(self) -> "_NumpyMLP":
        cp = _NumpyMLP.__new__(_NumpyMLP)
        cp.params = {k: v.copy() for k, v in self.params.items()}
        return cp

    def quantize_int8_in_place(self) -> None:
        for k in ("W1", "W2"):
            w = self.params[k]
            scale = np.maximum(np.max(np.abs(w), axis=0, keepdims=True), 1e-8) / 127.0
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            self.params[k] = (q.astype(np.float32) * scale).astype(np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        h = _np_relu(x @ self.params["W1"] + self.params["b1"])
        logits = h @ self.params["W2"] + self.params["b2"]
        return logits, h

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return logits

    def train_epochs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float = 1e-4,
        prox_mu: float = 0.0,
        global_ref: Optional[Dict[str, np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(42)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n = len(X)
        if n == 0:
            return
        num_classes = self.params["b2"].shape[1]

        for _ in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X[idx]
                yb = y[idx]
                bs = len(idx)

                logits, h = self.forward(xb)
                probs = _np_softmax(logits)
                y_oh = np.eye(num_classes, dtype=np.float32)[yb]
                dlogits = (probs - y_oh) / float(bs)

                dW2 = h.T @ dlogits + weight_decay * self.params["W2"]
                db2 = np.sum(dlogits, axis=0, keepdims=True)
                dh = dlogits @ self.params["W2"].T
                dh[h <= 0] = 0.0
                dW1 = xb.T @ dh + weight_decay * self.params["W1"]
                db1 = np.sum(dh, axis=0, keepdims=True)

                if prox_mu > 0.0 and global_ref is not None:
                    dW1 = dW1 + prox_mu * (self.params["W1"] - global_ref["W1"])
                    db1 = db1 + prox_mu * (self.params["b1"] - global_ref["b1"])
                    dW2 = dW2 + prox_mu * (self.params["W2"] - global_ref["W2"])
                    db2 = db2 + prox_mu * (self.params["b2"] - global_ref["b2"])

                self.params["W1"] -= lr * dW1
                self.params["b1"] -= lr * db1
                self.params["W2"] -= lr * dW2
                self.params["b2"] -= lr * db2


# ============================================================================
# PyTorch Helper Functions (Used when PyTorch is available on Local / `rama`)
# ============================================================================


if TORCH_AVAILABLE:

    def _make_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y).long()
        )
        # BatchNorm1d requires batch_size > 1 during training if drop_last is needed
        drop_last = shuffle and (len(ds) > batch_size) and (len(ds) % batch_size == 1)
        return DataLoader(
            ds,
            batch_size=min(batch_size, max(1, len(ds))),
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @torch.no_grad()
    def _eval_torch_model_logits(
        model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256
    ) -> np.ndarray:
        model.eval()
        model.to(device)
        out_list = []
        n = len(X)
        for start in range(0, n, batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            logits = model(xb)
            out_list.append(logits.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _train_torch_model(
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        use_bf16: bool = False,
        prox_mu: float = 0.0,
        global_params_ref: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        if len(X) == 0:
            return
        model.to(device)
        model.train()
        # If client only has 1 sample in batch, put BatchNorm layers in eval mode
        if len(X) == 1:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        loader = _make_loader(X, y, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        bf16_ok = use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

        for _ in range(epochs):
            for xb, yb in loader:
                if len(xb) == 1:
                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.eval()
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                if bf16_ok:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)

                if prox_mu > 0.0 and global_params_ref is not None:
                    prox = torch.tensor(0.0, device=device)
                    for name, p in model.named_parameters():
                        prox = prox + torch.sum((p - global_params_ref[name]) ** 2)
                    loss = loss + 0.5 * prox_mu * prox

                loss.backward()
                optimizer.step()

    def _quantize_torch_model_int8(model: nn.Module, device: torch.device) -> nn.Module:
        q_model = copy.deepcopy(model).cpu()
        state = q_model.state_dict()
        for k, v in state.items():
            if v.is_floating_point() and v.ndim >= 2 and v.numel() >= 16:
                flat = v.view(v.size(0), -1)
                scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                shape = [v.size(0)] + [1] * (v.ndim - 1)
                scale_b = scale.view(*shape)
                q = torch.clamp(torch.round(v / scale_b), -127, 127).to(torch.int8)
                state[k] = q.to(v.dtype) * scale_b
        q_model.load_state_dict(state)
        return q_model.to(device)


# ============================================================================
# 1. Centralized Upper-Bound Engine
# ============================================================================


def run_centralized_upper_bound(
    bundle: LoadedDatasetBundle,
    model_cfg: ModelConfig,
    bench_cfg: BenchmarkConfig,
    hw_cfg: HardwareConfig,
) -> AlgorithmRunResult:
    """
    Trains the Mode 1 unified model on 100% of pooled training data `X_train, y_train`.
    Establishes the theoretical Upper Bound (`A_cent`).
    """
    t0 = time.perf_counter()
    hidden_dim = (model_cfg.hidden_dims or [128, 64])[0]

    if TORCH_AVAILABLE:
        device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(bench_cfg.seed)
        model = build_torch_client_model(bundle, model_cfg).to(device)
        _train_torch_model(
            model,
            bundle.X_train,
            bundle.y_train,
            epochs=bench_cfg.centralized_epochs,
            batch_size=bench_cfg.batch_size,
            lr=bench_cfg.client_lr * 2.0,
            device=device,
            use_bf16=hw_cfg.use_bf16,
        )
        val_logits = _eval_torch_model_logits(model, bundle.X_val, device)
        test_logits = _eval_torch_model_logits(model, bundle.X_test, device)
    else:
        model_np = _NumpyMLP(
            bundle.input_dim, hidden_dim, bundle.num_classes, seed=bench_cfg.seed
        )
        model_np.train_epochs(
            bundle.X_train,
            bundle.y_train,
            epochs=bench_cfg.centralized_epochs,
            batch_size=bench_cfg.batch_size,
            lr=bench_cfg.client_lr * 3.0,
        )
        val_logits = model_np.predict_logits(bundle.X_val)
        test_logits = model_np.predict_logits(bundle.X_test)

    val_metrics = compute_classification_metrics(
        bundle.y_val, val_logits.argmax(axis=1), bundle.num_classes
    )
    test_metrics = compute_classification_metrics(
        bundle.y_test, test_logits.argmax(axis=1), bundle.num_classes
    )
    elapsed = time.perf_counter() - t0

    return AlgorithmRunResult(
        name="Centralized (Pooled Upper Bound)",
        category="centralized",
        test_metrics=test_metrics,
        val_accuracy=val_metrics.accuracy,
        best_round=bench_cfg.centralized_epochs,
        total_rounds=bench_cfg.centralized_epochs,
        comm_bytes=0,
        wall_time_sec=elapsed,
    )


# ============================================================================
# 2. Isolated Local Silos Lower-Bound Engine
# ============================================================================


def run_isolated_local_silos(
    bundle: LoadedDatasetBundle,
    partition: PartitionBundle,
    model_cfg: ModelConfig,
    bench_cfg: BenchmarkConfig,
    hw_cfg: HardwareConfig,
) -> IsolatedSilosResult:
    """
    Trains each client i in {0, ..., M-1} strictly on its own local data D_i
    with zero federation. Establishes the Isolated Silo Lower Bound (`A_local`).
    """
    t0 = time.perf_counter()
    hidden_dim = (model_cfg.hidden_dims or [128, 64])[0]
    m = partition.num_clients

    client_accs: List[float] = []
    client_f1s: List[float] = []
    client_samples: List[int] = []

    for cid in range(m):
        idxs = partition.client_full_indices[cid]
        X_c = bundle.X_train[idxs]
        y_c = bundle.y_train[idxs]
        client_samples.append(len(idxs))

        if TORCH_AVAILABLE:
            device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
            torch.manual_seed(bench_cfg.seed + cid)
            c_model = build_torch_client_model(bundle, model_cfg).to(device)
            _train_torch_model(
                c_model,
                X_c,
                y_c,
                epochs=bench_cfg.isolated_epochs,
                batch_size=bench_cfg.batch_size,
                lr=bench_cfg.client_lr,
                device=device,
                use_bf16=hw_cfg.use_bf16,
            )
            test_logits = _eval_torch_model_logits(c_model, bundle.X_test, device)
        else:
            c_model_np = _NumpyMLP(
                bundle.input_dim, hidden_dim, bundle.num_classes, seed=bench_cfg.seed + cid
            )
            c_model_np.train_epochs(
                X_c,
                y_c,
                epochs=bench_cfg.isolated_epochs,
                batch_size=bench_cfg.batch_size,
                lr=bench_cfg.client_lr * 2.0,
            )
            test_logits = c_model_np.predict_logits(bundle.X_test)

        metrics = compute_classification_metrics(
            bundle.y_test, test_logits.argmax(axis=1), bundle.num_classes
        )
        client_accs.append(metrics.accuracy)
        client_f1s.append(metrics.macro_f1)

    weights = np.array(client_samples, dtype=np.float64)
    weights = weights / np.maximum(1.0, weights.sum())
    w_acc = float(np.sum(weights * np.array(client_accs)))
    w_f1 = float(np.sum(weights * np.array(client_f1s)))
    elapsed = time.perf_counter() - t0

    return IsolatedSilosResult(
        weighted_test_accuracy=w_acc,
        weighted_test_macro_f1=w_f1,
        unweighted_mean_accuracy=float(np.mean(client_accs)),
        min_client_accuracy=float(np.min(client_accs)),
        max_client_accuracy=float(np.max(client_accs)),
        std_client_accuracy=float(np.std(client_accs)),
        per_client_accuracy=client_accs,
        per_client_macro_f1=client_f1s,
        per_client_samples=client_samples,
        wall_time_sec=elapsed,
    )


# ============================================================================
# 3. Iterative Federated Learning Engine (FedAvg, FedProx, FedAdam)
# ============================================================================


def run_iterative_fl(
    algorithm: str,
    bundle: LoadedDatasetBundle,
    partition: PartitionBundle,
    model_spec: ModelSpec,
    model_cfg: ModelConfig,
    bench_cfg: BenchmarkConfig,
    hw_cfg: HardwareConfig,
) -> AlgorithmRunResult:
    """
    Runs Mode 1 Iterative Federated Learning (`fedavg`, `fedprox`, or `fedadam`).
    """
    algo = algorithm.lower()
    display_name = {
        "fedavg": "FedAvg (Iterative)",
        "fedprox": "FedProx (Iterative)",
        "fedadam": "FedAdam (Iterative)",
    }.get(algo, algo.upper())

    t0 = time.perf_counter()
    m = partition.num_clients
    weights = np.array(partition.client_sample_totals, dtype=np.float64)
    weights = weights / np.maximum(1.0, weights.sum())

    comm_per_round = 2 * m * model_spec.fp32_size_bytes
    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_test_metrics: Optional[ClassificationMetrics] = None
    best_round = 1

    if TORCH_AVAILABLE:
        device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(bench_cfg.seed)
        global_model = build_torch_client_model(bundle, model_cfg).to(device)
        local_model = copy.deepcopy(global_model).to(device)

        server_m = {
            k: torch.zeros_like(p.data) for k, p in global_model.named_parameters()
        }
        server_v = {
            k: torch.zeros_like(p.data) for k, p in global_model.named_parameters()
        }

        for rnd in range(1, bench_cfg.fl_global_rounds + 1):
            g_state = {
                k: v.detach().clone() for k, v in global_model.state_dict().items()
            }
            g_params_ref = {
                k: p.detach().clone() for k, p in global_model.named_parameters()
            }
            accum = {
                k: torch.zeros_like(v, dtype=torch.float32)
                for k, v in g_state.items()
                if v.is_floating_point()
            }

            for cid in range(m):
                local_model.load_state_dict(g_state)
                idxs = partition.client_full_indices[cid]
                _train_torch_model(
                    local_model,
                    bundle.X_train[idxs],
                    bundle.y_train[idxs],
                    epochs=bench_cfg.fl_local_epochs,
                    batch_size=bench_cfg.batch_size,
                    lr=bench_cfg.client_lr,
                    device=device,
                    use_bf16=hw_cfg.use_bf16,
                    prox_mu=(bench_cfg.prox_mu if algo == "fedprox" else 0.0),
                    global_params_ref=(g_params_ref if algo == "fedprox" else None),
                )
                w_i = float(weights[cid])
                with torch.no_grad():
                    for k, v in local_model.state_dict().items():
                        if k in accum:
                            accum[k].add_(v.to(torch.float32), alpha=w_i)

            with torch.no_grad():
                if algo in ("fedavg", "fedprox"):
                    new_state = copy.deepcopy(g_state)
                    for k, v in accum.items():
                        new_state[k].copy_(v.to(g_state[k].dtype))
                    global_model.load_state_dict(new_state)
                else:  # fedadam
                    new_state = copy.deepcopy(g_state)
                    for k, v in accum.items():
                        if k not in server_m:
                            new_state[k].copy_(v.to(g_state[k].dtype))
                    global_model.load_state_dict(new_state)
                    for name, p in global_model.named_parameters():
                        delta = accum[name].to(p.dtype) - g_state[name]
                        server_m[name].mul_(0.9).add_(delta, alpha=0.1)
                        server_v[name].mul_(0.99).addcmul_(delta, delta, value=0.01)
                        p.data.addcdiv_(
                            server_m[name],
                            torch.sqrt(server_v[name].clamp(min=1e-12)) + 1e-3,
                            value=bench_cfg.server_lr,
                        )

            val_logits = _eval_torch_model_logits(global_model, bundle.X_val, device)
            test_logits = _eval_torch_model_logits(global_model, bundle.X_test, device)
            val_m = compute_classification_metrics(
                bundle.y_val, val_logits.argmax(axis=1), bundle.num_classes
            )
            test_m = compute_classification_metrics(
                bundle.y_test, test_logits.argmax(axis=1), bundle.num_classes
            )
            history.append(
                {
                    "round": float(rnd),
                    "val_acc": round(val_m.accuracy, 2),
                    "test_acc": round(test_m.accuracy, 2),
                    "test_macro_f1": round(test_m.macro_f1, 2),
                }
            )
            if val_m.accuracy >= best_val_acc:
                best_val_acc = val_m.accuracy
                best_test_metrics = test_m
                best_round = rnd

    else:
        # Pure-NumPy Iterative FL execution
        hidden_dim = (model_cfg.hidden_dims or [128, 64])[0]
        global_np = _NumpyMLP(
            bundle.input_dim, hidden_dim, bundle.num_classes, seed=bench_cfg.seed
        )
        server_m_np = {k: np.zeros_like(v) for k, v in global_np.params.items()}
        server_v_np = {k: np.zeros_like(v) for k, v in global_np.params.items()}
        rng = np.random.default_rng(bench_cfg.seed)

        for rnd in range(1, bench_cfg.fl_global_rounds + 1):
            g_params = {k: v.copy() for k, v in global_np.params.items()}
            accum_np = {k: np.zeros_like(v) for k, v in g_params.items()}

            for cid in range(m):
                c_np = global_np.clone()
                idxs = partition.client_full_indices[cid]
                c_np.train_epochs(
                    bundle.X_train[idxs],
                    bundle.y_train[idxs],
                    epochs=bench_cfg.fl_local_epochs,
                    batch_size=bench_cfg.batch_size,
                    lr=bench_cfg.client_lr * 2.0,
                    prox_mu=(bench_cfg.prox_mu if algo == "fedprox" else 0.0),
                    global_ref=(g_params if algo == "fedprox" else None),
                    rng=rng,
                )
                w_i = float(weights[cid])
                for k in accum_np:
                    accum_np[k] += w_i * c_np.params[k]

            if algo in ("fedavg", "fedprox"):
                global_np.params = accum_np
            else:
                for k in global_np.params:
                    delta = accum_np[k] - g_params[k]
                    server_m_np[k] = 0.9 * server_m_np[k] + 0.1 * delta
                    server_v_np[k] = 0.99 * server_v_np[k] + 0.01 * (delta * delta)
                    global_np.params[k] = g_params[k] + bench_cfg.server_lr * (
                        server_m_np[k] / (np.sqrt(server_v_np[k]) + 1e-3)
                    )

            val_logits = global_np.predict_logits(bundle.X_val)
            test_logits = global_np.predict_logits(bundle.X_test)
            val_m = compute_classification_metrics(
                bundle.y_val, val_logits.argmax(axis=1), bundle.num_classes
            )
            test_m = compute_classification_metrics(
                bundle.y_test, test_logits.argmax(axis=1), bundle.num_classes
            )
            history.append(
                {
                    "round": float(rnd),
                    "val_acc": round(val_m.accuracy, 2),
                    "test_acc": round(test_m.accuracy, 2),
                    "test_macro_f1": round(test_m.macro_f1, 2),
                }
            )
            if val_m.accuracy >= best_val_acc:
                best_val_acc = val_m.accuracy
                best_test_metrics = test_m
                best_round = rnd

    assert best_test_metrics is not None
    elapsed = time.perf_counter() - t0
    return AlgorithmRunResult(
        name=display_name,
        category="iterative_fl",
        test_metrics=best_test_metrics,
        val_accuracy=best_val_acc,
        best_round=best_round,
        total_rounds=bench_cfg.fl_global_rounds,
        comm_bytes=best_round * comm_per_round,
        wall_time_sec=elapsed,
        round_history=history,
    )


# ============================================================================
# 4. FENS Two-Phase Ensembling Engine (NeurIPS 2024)
# ============================================================================


def run_fens_fl(
    bundle: LoadedDatasetBundle,
    partition: PartitionBundle,
    model_spec: ModelSpec,
    model_cfg: ModelConfig,
    bench_cfg: BenchmarkConfig,
    hw_cfg: HardwareConfig,
) -> Tuple[AlgorithmRunResult, AlgorithmRunResult]:
    """
    Executes both:
      1. `FedEns (One-Shot Uniform Averaging)`
      2. `FENS (One-Shot + INT8 Quantization + Phase 2 FedAdam Aggregator)`
    on the same Phase 1 locally trained client models!
    """
    t0 = time.perf_counter()
    m = partition.num_clients
    c = bundle.num_classes
    hidden_dim = (model_cfg.hidden_dims or [128, 64])[0]

    # Phase 1: Train M local models on their 90% D_i1 splits and compute logits on D_i2, Val, Test
    d2_concat_logits: Dict[int, np.ndarray] = {}
    d2_labels: Dict[int, np.ndarray] = {}

    if TORCH_AVAILABLE:
        device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(bench_cfg.seed)
        client_models = []
        for cid in range(m):
            c_model = build_torch_client_model(bundle, model_cfg).to(device)
            d1_idx = partition.client_d1_indices[cid]
            _train_torch_model(
                c_model,
                bundle.X_train[d1_idx],
                bundle.y_train[d1_idx],
                epochs=bench_cfg.fens_phase1_epochs,
                batch_size=bench_cfg.batch_size,
                lr=bench_cfg.client_lr * 1.5,
                device=device,
                use_bf16=hw_cfg.use_bf16,
            )
            if bench_cfg.fens_quantize_int8:
                c_model = _quantize_torch_model_int8(c_model, device)
            client_models.append(c_model)

        for cid in range(m):
            d2_idx = partition.client_d2_indices[cid]
            X_d2 = bundle.X_train[d2_idx]
            per_m = [_eval_torch_model_logits(cm, X_d2, device) for cm in client_models]
            d2_concat_logits[cid] = np.concatenate(per_m, axis=1)
            d2_labels[cid] = bundle.y_train[d2_idx]

        val_concat = np.concatenate(
            [_eval_torch_model_logits(cm, bundle.X_val, device) for cm in client_models],
            axis=1,
        )
        test_concat = np.concatenate(
            [_eval_torch_model_logits(cm, bundle.X_test, device) for cm in client_models],
            axis=1,
        )

    else:
        client_models_np = []
        for cid in range(m):
            c_np = _NumpyMLP(
                bundle.input_dim, hidden_dim, c, seed=bench_cfg.seed + cid
            )
            d1_idx = partition.client_d1_indices[cid]
            c_np.train_epochs(
                bundle.X_train[d1_idx],
                bundle.y_train[d1_idx],
                epochs=bench_cfg.fens_phase1_epochs,
                batch_size=bench_cfg.batch_size,
                lr=bench_cfg.client_lr * 2.5,
            )
            if bench_cfg.fens_quantize_int8:
                c_np.quantize_int8_in_place()
            client_models_np.append(c_np)

        for cid in range(m):
            d2_idx = partition.client_d2_indices[cid]
            X_d2 = bundle.X_train[d2_idx]
            d2_concat_logits[cid] = np.concatenate(
                [cm.predict_logits(X_d2) for cm in client_models_np], axis=1
            )
            d2_labels[cid] = bundle.y_train[d2_idx]

        val_concat = np.concatenate(
            [cm.predict_logits(bundle.X_val) for cm in client_models_np], axis=1
        )
        test_concat = np.concatenate(
            [cm.predict_logits(bundle.X_test) for cm in client_models_np], axis=1
        )

    t_phase1 = time.perf_counter() - t0
    ofl_comm_bytes = 2 * m * model_spec.fp32_size_bytes

    # Evaluate One-Shot FedEns (Uniform Averaging)
    val_fedens_logits = val_concat.reshape(len(bundle.X_val), m, c).mean(axis=1)
    test_fedens_logits = test_concat.reshape(len(bundle.X_test), m, c).mean(axis=1)
    fedens_val_m = compute_classification_metrics(
        bundle.y_val, val_fedens_logits.argmax(axis=1), c
    )
    fedens_test_m = compute_classification_metrics(
        bundle.y_test, test_fedens_logits.argmax(axis=1), c
    )
    fedens_result = AlgorithmRunResult(
        name="FedEns (One-Shot Uniform)",
        category="one_shot_fl",
        test_metrics=fedens_test_metrics if (fedens_test_metrics := fedens_test_m) else fedens_test_m,
        val_accuracy=fedens_val_m.accuracy,
        best_round=1,
        total_rounds=1,
        comm_bytes=ofl_comm_bytes,
        wall_time_sec=t_phase1,
        round_history=[
            {
                "round": 1.0,
                "val_acc": round(fedens_val_m.accuracy, 2),
                "test_acc": round(fedens_test_m.accuracy, 2),
                "test_macro_f1": round(fedens_test_m.macro_f1, 2),
            }
        ],
    )

    # Phase 2: Train FENS 2-Layer ReLU MLP Aggregator f_lambda(z_1, ..., z_M) via FedAdam
    t_p2_start = time.perf_counter()
    agg_k = bench_cfg.fens_hidden_dim
    agg_np = _NumpyMLP(m * c, agg_k, c, seed=bench_cfg.seed + 999)
    agg_param_bytes = ((m * c) * agg_k + agg_k + agg_k * c + c) * 4

    server_m_agg = {k: np.zeros_like(v) for k, v in agg_np.params.items()}
    server_v_agg = {k: np.zeros_like(v) for k, v in agg_np.params.items()}
    rng = np.random.default_rng(bench_cfg.seed)

    best_val_acc = -1.0
    best_test_m: Optional[ClassificationMetrics] = None
    best_round = 1
    history: List[Dict[str, float]] = []

    eval_every = max(1, bench_cfg.fens_phase2_rounds // 25)
    for rnd in range(1, bench_cfg.fens_phase2_rounds + 1):
        g_agg = {k: v.copy() for k, v in agg_np.params.items()}
        delta_sum = {k: np.zeros_like(v) for k, v in g_agg.items()}

        for cid in range(m):
            c_agg = agg_np.clone()
            c_agg.train_epochs(
                d2_concat_logits[cid],
                d2_labels[cid],
                epochs=1,
                batch_size=min(128, max(1, len(d2_labels[cid]))),
                lr=bench_cfg.fens_client_lr,
                weight_decay=1e-4,
                rng=rng,
            )
            for k in delta_sum:
                delta_sum[k] += (c_agg.params[k] - g_agg[k]) / float(m)

        for k in agg_np.params:
            d_avg = delta_sum[k]
            server_m_agg[k] = 0.9 * server_m_agg[k] + 0.1 * d_avg
            server_v_agg[k] = 0.99 * server_v_agg[k] + 0.01 * (d_avg * d_avg)
            agg_np.params[k] = g_agg[k] + bench_cfg.fens_server_lr * (
                server_m_agg[k] / (np.sqrt(server_v_agg[k]) + 1e-3)
            )

        if rnd == 1 or rnd % eval_every == 0 or rnd == bench_cfg.fens_phase2_rounds:
            v_logits = agg_np.predict_logits(val_concat)
            t_logits = agg_np.predict_logits(test_concat)
            v_m = compute_classification_metrics(
                bundle.y_val, v_logits.argmax(axis=1), c
            )
            t_m = compute_classification_metrics(
                bundle.y_test, t_logits.argmax(axis=1), c
            )
            history.append(
                {
                    "round": float(rnd),
                    "val_acc": round(v_m.accuracy, 2),
                    "test_acc": round(t_m.accuracy, 2),
                    "test_macro_f1": round(t_m.macro_f1, 2),
                }
            )
            if v_m.accuracy >= best_val_acc:
                best_val_acc = v_m.accuracy
                best_test_m = t_m
                best_round = rnd

    assert best_test_m is not None
    dl_bytes = (
        m * (m - 1) * model_spec.int8_size_bytes
        if bench_cfg.fens_quantize_int8
        else m * (m - 1) * model_spec.fp32_size_bytes
    )
    p2_comm_bytes = 2 * m * best_round * agg_param_bytes
    total_fens_comm = ofl_comm_bytes + dl_bytes + p2_comm_bytes
    total_fens_time = t_phase1 + (time.perf_counter() - t_p2_start)

    fens_result = AlgorithmRunResult(
        name="FENS (Two-Phase Ensembling)",
        category="one_shot_fl",
        test_metrics=best_test_m,
        val_accuracy=best_val_acc,
        best_round=best_round,
        total_rounds=bench_cfg.fens_phase2_rounds,
        comm_bytes=total_fens_comm,
        wall_time_sec=total_fens_time,
        round_history=history,
        extra_info={
            "phase1_upload_mib": round(ofl_comm_bytes / (1024.0 ** 2), 3),
            "int8_download_mib": round(dl_bytes / (1024.0 ** 2), 3),
            "phase2_agg_mib": round(p2_comm_bytes / (1024.0 ** 2), 3),
            "normalized_comm_wrt_ofl": round(total_fens_comm / max(1, ofl_comm_bytes), 2),
        },
    )

    return fedens_result, fens_result
