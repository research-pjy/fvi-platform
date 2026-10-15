"""
Federated Client Partitioning Engine & Heterogeneity Diagnostics.

Supports 4 partitioning strategies:
  1. `iid`: Uniform stratified assignment across M clients.
  2. `dirichlet`: Non-IID label skew via Dirichlet distribution Dir_M(alpha).
  3. `quantity_skew`: Unequal sample volume across clients (power-law / Dirichlet
     volume proportions) with uniform class probabilities.
  4. `generalist_specialist`: Client 0 is a balanced generalist across all classes,
     while the remaining clients specialize in 1-2 classes (`specialist_classes`).

Also computes per-client normalized Shannon entropy H_i in [0, 1] (where 1.0 means
perfectly balanced across all C classes and 0.0 means single-class only) and the
90% (D_i1) / 10% (D_i2) local split required by FENS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from fvi.config import PartitionConfig


@dataclass
class PartitionBundle:
    """Stores client index partitions and statistical heterogeneity diagnostics."""

    num_clients: int
    num_classes: int
    strategy: str
    dirichlet_alpha: float
    client_full_indices: Dict[int, np.ndarray]
    client_d1_indices: Dict[int, np.ndarray]
    client_d2_indices: Dict[int, np.ndarray]
    class_counts_full: np.ndarray  # shape (M, C)
    class_counts_d1: np.ndarray    # shape (M, C)
    class_counts_d2: np.ndarray    # shape (M, C)
    client_sample_totals: List[int]
    client_normalized_entropy: List[float]  # in [0, 1]
    mean_normalized_entropy: float
    description: str

    def to_summary_dict(self, class_names: Sequence[str]) -> Dict[str, Any]:
        return {
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "class_names": list(class_names),
            "strategy": self.strategy,
            "dirichlet_alpha": self.dirichlet_alpha,
            "description": self.description,
            "class_counts_full": self.class_counts_full.tolist(),
            "client_sample_totals": self.client_sample_totals,
            "client_normalized_entropy": [
                round(e, 3) for e in self.client_normalized_entropy
            ],
            "mean_normalized_entropy": round(self.mean_normalized_entropy, 3),
        }


def _compute_counts(
    y: np.ndarray,
    client_indices: Dict[int, np.ndarray],
    num_clients: int,
    num_classes: int,
) -> np.ndarray:
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for cid, idxs in client_indices.items():
        if len(idxs) > 0:
            labels_c = y[idxs]
            for c in range(num_classes):
                counts[cid, c] = int(np.sum(labels_c == c))
    return counts


def _compute_normalized_entropy(counts: np.ndarray) -> List[float]:
    """
    Computes normalized Shannon label entropy H_i / log(C) in [0, 1] for each client.
    """
    num_clients, num_classes = counts.shape
    if num_classes <= 1:
        return [1.0] * num_clients
    max_h = np.log(float(num_classes))
    entropies: List[float] = []
    for i in range(num_clients):
        tot = float(counts[i].sum())
        if tot <= 0:
            entropies.append(0.0)
            continue
        probs = counts[i] / tot
        probs = probs[probs > 0]
        h = -float(np.sum(probs * np.log(probs)))
        entropies.append(float(np.clip(h / max_h, 0.0, 1.0)))
    return entropies


def _split_d1_d2(
    client_full: Dict[int, np.ndarray],
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    d1: Dict[int, np.ndarray] = {}
    d2: Dict[int, np.ndarray] = {}
    for cid, idxs in client_full.items():
        shuffled = rng.permutation(idxs)
        n = len(shuffled)
        n_d1 = max(1, int(round(n * train_fraction)))
        if n_d1 >= n and n > 1:
            n_d1 = n - 1
        d1[cid] = shuffled[:n_d1]
        d2[cid] = shuffled[n_d1:] if n > 1 else shuffled[:1]
    return d1, d2


def partition_dataset(
    y_train: np.ndarray,
    cfg: PartitionConfig,
    num_classes: int,
) -> PartitionBundle:
    """
    Partitions training indices `0..len(y_train)-1` across `cfg.num_clients`
    according to `cfg.strategy`.
    """
    y = np.asarray(y_train, dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    m = max(2, int(cfg.num_clients))
    strat = cfg.strategy.lower()

    client_lists: List[List[int]] = [[] for _ in range(m)]

    if strat == "iid":
        for c in range(num_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            splits = np.array_split(idx_c, m)
            for cid, sp in enumerate(splits):
                client_lists[cid].extend(sp.tolist())
        desc = f"IID Uniform Split across M={m} Clients"

    elif strat == "quantity_skew":
        all_idx = np.arange(len(y), dtype=np.int64)
        rng.shuffle(all_idx)
        proportions = rng.dirichlet(np.repeat(max(0.25, cfg.dirichlet_alpha), m))
        # Ensure minimum 10 samples per client
        min_frac = min(0.5 / m, 10.0 / max(100, len(y)))
        proportions = np.maximum(proportions, min_frac)
        proportions = proportions / proportions.sum()
        split_pts = (np.cumsum(proportions) * len(all_idx)).astype(int)[:-1]
        for cid, sp in enumerate(np.split(all_idx, split_pts)):
            client_lists[cid].extend(sp.tolist())
        desc = f"Quantity Skew (alpha={cfg.dirichlet_alpha}) across M={m} Clients"

    elif strat == "generalist_specialist":
        spec_set = set(int(c) % num_classes for c in (cfg.specialist_classes or [0, 1]))
        for c in range(num_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            if c in spec_set:
                # Client 0 gets 1/(m+1) share, specialist clients 1..m-1 share the rest
                splits = np.array_split(idx_c, m)
                for cid, sp in enumerate(splits):
                    client_lists[cid].extend(sp.tolist())
            else:
                # Non-specialist classes go entirely to Client 0 (Generalist)
                client_lists[0].extend(idx_c.tolist())
        spec_str = ", ".join(str(s) for s in sorted(spec_set))
        desc = f"Generalist (Client 0: All {num_classes} Classes) vs. Specialists (Clients 1..{m-1}: Classes {{{spec_str}}})"

    else:  # Default: Dirichlet Dir_M(alpha) label skew
        alpha = max(0.005, float(cfg.dirichlet_alpha))
        min_size = 0
        attempt = 0
        min_required = min(8, max(2, len(y) // (m * 4)))

        while min_size < min_required:
            attempt += 1
            client_lists = [[] for _ in range(m)]
            for c in range(num_classes):
                idx_c = np.where(y == c)[0]
                rng.shuffle(idx_c)
                proportions = rng.dirichlet(np.repeat(alpha, m))
                max_cap = len(y) / m
                proportions = np.array(
                    [
                        p * (len(c_list) < max_cap)
                        for p, c_list in zip(proportions, client_lists)
                    ]
                )
                if proportions.sum() <= 0:
                    proportions = np.ones(m) / m
                else:
                    proportions = proportions / proportions.sum()
                split_pts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                for cid, sp in enumerate(np.split(idx_c, split_pts)):
                    client_lists[cid].extend(sp.tolist())

            min_size = min(len(c_list) for c_list in client_lists)
            if attempt > 150 and min_size >= 2:
                break

        # Fallback guarantee that no client is empty
        for cid in range(m):
            if len(client_lists[cid]) < 2:
                largest_cid = int(np.argmax([len(lst) for lst in client_lists]))
                donated = client_lists[largest_cid][:2]
                client_lists[largest_cid] = client_lists[largest_cid][2:]
                client_lists[cid].extend(donated)

        desc = f"Non-IID Dirichlet Dir_{m}(alpha={alpha})"

    client_full = {
        cid: np.array(client_lists[cid], dtype=np.int64) for cid in range(m)
    }
    client_d1, client_d2 = _split_d1_d2(client_full, cfg.fens_train_fraction, rng)

    counts_full = _compute_counts(y, client_full, m, num_classes)
    counts_d1 = _compute_counts(y, client_d1, m, num_classes)
    counts_d2 = _compute_counts(y, client_d2, m, num_classes)
    totals = [int(counts_full[i].sum()) for i in range(m)]
    entropies = _compute_normalized_entropy(counts_full)

    return PartitionBundle(
        num_clients=m,
        num_classes=num_classes,
        strategy=strat,
        dirichlet_alpha=cfg.dirichlet_alpha,
        client_full_indices=client_full,
        client_d1_indices=client_d1,
        client_d2_indices=client_d2,
        class_counts_full=counts_full,
        class_counts_d1=counts_d1,
        class_counts_d2=counts_d2,
        client_sample_totals=totals,
        client_normalized_entropy=entropies,
        mean_normalized_entropy=float(np.mean(entropies)),
        description=desc,
    )
