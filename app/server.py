#!/usr/bin/env python3
"""
FastAPI Web Server & Zero-Disk In-RAM REST API for FedAssess (`fvi-platform`).

Key Privacy & Storage Guarantee:
  - Dropped datasets from the Web UI are parsed/resized locally in the user's
    browser and either:
      (Mode A) Trained 100% inside the user's browser tab (0 bytes sent to server), OR
      (Mode B) Sent as a compact numerical tensor to `POST /api/run-in-memory-benchmark`,
               where `rama` holds the tensor purely in RAM during L40S GPU training
               and immediately garbage-collects it upon completion—writing ZERO
               uploaded files to `rama`'s hard drive!
"""

from __future__ import annotations

import argparse
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from fvi.config import (
    BenchmarkConfig,
    DatasetConfig,
    FVJobConfig,
    HardwareConfig,
    ModelConfig,
    PartitionConfig,
)
from fvi.ingestion.loader import (
    LoadedDatasetBundle,
    bundle_from_in_memory_arrays,
    load_and_preprocess_dataset,
)
from fvi.ingestion.schema import inspect_dataset_source
from fvi.models.factory import TORCH_AVAILABLE
from fvi.pipeline import (
    preview_partition_from_bundle,
    preview_partition_only,
    run_fvi_benchmark_pipeline,
    run_fvi_on_loaded_bundle,
)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


STATIC_DIR = Path(__file__).resolve().parent / "static"

_JOB_LOCK = threading.Lock()
_JOB_STATE: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "message": "Idle",
    "error": None,
    "result": None,
}


def get_system_status() -> Dict[str, Any]:
    """Returns hardware, environment, and RAM/GPU status for the UI header."""
    hostname = socket.gethostname()
    conda_env = (
        os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("VIRTUAL_ENV")
        or "system"
    )
    if "/" in conda_env:
        conda_env = Path(conda_env).name

    gpu_name = "CPU"
    vram_gib = 0.0
    bf16_ok = False

    if TORCH_AVAILABLE:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            _, total_b = torch.cuda.mem_get_info(0)
            vram_gib = round(total_b / (1024.0 ** 3), 1)
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        else:
            gpu_name = "PyTorch CPU"

    return {
        "hostname": hostname,
        "conda_env": conda_env,
        "python_version": platform.python_version(),
        "web_framework": "FastAPI" if FASTAPI_AVAILABLE else "HTTP",
        "backend": "PyTorch" if TORCH_AVAILABLE else "NumPy",
        "gpu_name": gpu_name,
        "vram_gib": vram_gib,
        "bf16_supported": bf16_ok,
        "zero_disk_storage": True,
    }


def _parse_job_config(payload: Dict[str, Any], require_path: bool = True) -> FVJobConfig:
    ds_raw = payload.get("dataset", {})
    part_raw = payload.get("partition", {})
    mod_raw = payload.get("model", {})
    bench_raw = payload.get("benchmark", {})

    raw_path = str(ds_raw.get("data_path", "")).strip()
    if require_path and not raw_path:
        raise ValueError("No dataset path specified.")

    resolved_path = Path(raw_path) if raw_path else Path("in_memory")
    if raw_path and not resolved_path.is_absolute() and not resolved_path.exists():
        resolved_path = ROOT_DIR / raw_path

    ds_cfg = DatasetConfig(
        data_path=str(resolved_path),
        modality=str(ds_raw.get("modality", "tabular")),
        target_column=str(ds_raw.get("target_column", "target")),
        feature_columns=ds_raw.get("feature_columns", ["all"]),
        image_path_column=ds_raw.get("image_path_column", "image_path"),
        image_size=int(ds_raw.get("image_size", 32)),
        image_channels=int(ds_raw.get("image_channels", 1)),
        seed=int(ds_raw.get("seed", 42)),
    )

    spec_classes = part_raw.get("specialist_classes", [0, 1])
    if isinstance(spec_classes, str):
        spec_classes = [
            int(x.strip()) for x in spec_classes.split(",") if x.strip().isdigit()
        ]

    part_cfg = PartitionConfig(
        num_clients=int(part_raw.get("num_clients", 10)),
        strategy=str(part_raw.get("strategy", "dirichlet")),
        dirichlet_alpha=float(part_raw.get("dirichlet_alpha", 0.1)),
        specialist_classes=spec_classes or [0, 1],
        seed=int(part_raw.get("seed", 42)),
    )

    hidden_raw = mod_raw.get("hidden_dims", [128, 64])
    if isinstance(hidden_raw, str):
        hidden_raw = [
            int(x.strip()) for x in hidden_raw.split(",") if x.strip().isdigit()
        ]

    mod_cfg = ModelConfig(
        arch=str(mod_raw.get("arch", "auto")),
        hidden_dims=hidden_raw or [128, 64],
        dropout=float(mod_raw.get("dropout", 0.15)),
    )

    bench_cfg = BenchmarkConfig(
        algorithms=bench_raw.get(
            "algorithms", ["fedavg", "fedprox", "fedadam", "fens"]
        ),
        centralized_epochs=int(bench_raw.get("centralized_epochs", 20)),
        isolated_epochs=int(bench_raw.get("isolated_epochs", 20)),
        fl_global_rounds=int(bench_raw.get("fl_global_rounds", 20)),
        fl_local_epochs=int(bench_raw.get("fl_local_epochs", 2)),
        batch_size=int(bench_raw.get("batch_size", 32)),
        client_lr=float(bench_raw.get("client_lr", 0.01)),
        server_lr=float(bench_raw.get("server_lr", 0.01)),
        prox_mu=float(bench_raw.get("prox_mu", 0.05)),
        fens_phase1_epochs=int(bench_raw.get("fens_phase1_epochs", 20)),
        fens_phase2_rounds=int(bench_raw.get("fens_phase2_rounds", 120)),
        fens_quantize_int8=bool(bench_raw.get("fens_quantize_int8", True)),
        run_iid_reference_for_hr=bool(bench_raw.get("run_iid_reference_for_hr", True)),
        seed=int(bench_raw.get("seed", 42)),
    )

    return FVJobConfig(
        dataset=ds_cfg,
        partition=part_cfg,
        model=mod_cfg,
        benchmark=bench_cfg,
        hardware=HardwareConfig(),
        output_dir="",
    )


def _build_in_memory_bundle_from_payload(
    payload: Dict[str, Any]
) -> Tuple[LoadedDatasetBundle, FVJobConfig, str]:
    """
    Constructs a `LoadedDatasetBundle` purely in RAM from browser-preprocessed
    arrays (`X`, `y`, `class_names`, `feature_names`, `image_shape`).
    Zero files are ever written to disk.
    """
    mem_data = payload.get("in_memory_data")
    if not mem_data:
        raise ValueError("Missing 'in_memory_data' payload.")

    X_raw = np.asarray(mem_data["X"], dtype=np.float32)
    y_raw = np.asarray(mem_data["y"], dtype=np.int64)
    modality = str(mem_data.get("modality", "tabular"))
    class_names = list(mem_data.get("class_names", []))
    feature_names = list(mem_data.get("feature_names", []))
    img_shape_raw = mem_data.get("image_shape")
    image_shape = (
        (int(img_shape_raw[0]), int(img_shape_raw[1]), int(img_shape_raw[2]))
        if img_shape_raw and len(img_shape_raw) == 3
        else None
    )
    dataset_label = str(mem_data.get("dataset_name", "In-Memory Dataset"))

    job_cfg = _parse_job_config(payload, require_path=False)
    bundle = bundle_from_in_memory_arrays(
        X_raw=X_raw,
        y_raw=y_raw,
        modality=modality,
        class_names=class_names,
        feature_names=feature_names,
        image_shape=image_shape,
        seed=job_cfg.dataset.seed,
    )
    return bundle, job_cfg, dataset_label


def _background_in_memory_worker(
    bundle: LoadedDatasetBundle, job_cfg: FVJobConfig, dataset_label: str
) -> None:
    def _cb(pct: int, msg: str) -> None:
        with _JOB_LOCK:
            _JOB_STATE["progress"] = pct
            _JOB_STATE["message"] = msg

    try:
        res = run_fvi_on_loaded_bundle(
            bundle=bundle,
            job_cfg=job_cfg,
            dataset_label=dataset_label,
            progress_cb=_cb,
            save_artifacts_to_disk=False,  # Zero disk storage on rama!
        )
        with _JOB_LOCK:
            _JOB_STATE["running"] = False
            _JOB_STATE["progress"] = 100
            _JOB_STATE["message"] = "Completed (RAM buffer cleared)"
            _JOB_STATE["result"] = res
            _JOB_STATE["error"] = None
    except Exception as exc:
        traceback.print_exc()
        with _JOB_LOCK:
            _JOB_STATE["running"] = False
            _JOB_STATE["error"] = str(exc)
            _JOB_STATE["message"] = f"Error: {exc}"
    finally:
        # Explicitly release RAM tensors immediately after the run
        del bundle
        gc.collect()


if FASTAPI_AVAILABLE:
    fastapi_app = FastAPI(
        title="FedAssess — Federated Viability Index (FVI)",
        version="0.4.0",
    )

    @fastapi_app.get("/", response_class=HTMLResponse)
    async def serve_index() -> HTMLResponse:
        html_file = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))

    @fastapi_app.get("/api/system-status")
    async def api_system_status() -> JSONResponse:
        return JSONResponse(content=get_system_status())

    @fastapi_app.get("/api/job-status")
    async def api_job_status() -> JSONResponse:
        with _JOB_LOCK:
            return JSONResponse(content=dict(_JOB_STATE))

    @fastapi_app.post("/api/preview-in-memory-partition")
    async def api_preview_in_memory_partition(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            bundle, job_cfg, _ = _build_in_memory_bundle_from_payload(payload)
            preview_data = preview_partition_from_bundle(bundle, job_cfg)
            del bundle
            gc.collect()
            return JSONResponse(content=preview_data)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    @fastapi_app.post("/api/run-in-memory-benchmark")
    async def api_run_in_memory_benchmark(request: Request) -> JSONResponse:
        payload = await request.json()
        synchronous = bool(payload.get("synchronous", False))
        try:
            bundle, job_cfg, dataset_label = _build_in_memory_bundle_from_payload(payload)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

        if synchronous:
            res = run_fvi_on_loaded_bundle(
                bundle=bundle,
                job_cfg=job_cfg,
                dataset_label=dataset_label,
                save_artifacts_to_disk=False,
            )
            del bundle
            gc.collect()
            return JSONResponse(content={"status": "completed", "result": res})

        with _JOB_LOCK:
            if _JOB_STATE["running"]:
                return JSONResponse(
                    content={"error": "A benchmark is currently in progress."},
                    status_code=409,
                )
            _JOB_STATE["running"] = True
            _JOB_STATE["progress"] = 4
            _JOB_STATE["message"] = "Running ephemeral in-RAM evaluation on server..."
            _JOB_STATE["error"] = None
            _JOB_STATE["result"] = None

        t = threading.Thread(
            target=_background_in_memory_worker,
            args=(bundle, job_cfg, dataset_label),
            daemon=True,
        )
        t.start()
        return JSONResponse(content={"status": "started"})


class FVIRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/job-status" not in (args[0] if args else ""):
            sys.stderr.write(f"[FVI-Server] {fmt % args}\n")

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            content = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if route == "/api/system-status":
            self._send_json(get_system_status())
            return
        if route == "/api/job-status":
            with _JOB_LOCK:
                self._send_json(dict(_JOB_STATE))
            return
        self._send_json({"error": f"Not found: {route}"}, status=404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            payload = self._read_json_body()
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return

        if route == "/api/preview-in-memory-partition":
            try:
                bundle, job_cfg, _ = _build_in_memory_bundle_from_payload(payload)
                prev = preview_partition_from_bundle(bundle, job_cfg)
                del bundle
                gc.collect()
                self._send_json(prev)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if route == "/api/run-in-memory-benchmark":
            synchronous = bool(payload.get("synchronous", False))
            try:
                bundle, job_cfg, dataset_label = _build_in_memory_bundle_from_payload(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            if synchronous:
                res = run_fvi_on_loaded_bundle(
                    bundle=bundle,
                    job_cfg=job_cfg,
                    dataset_label=dataset_label,
                    save_artifacts_to_disk=False,
                )
                del bundle
                gc.collect()
                self._send_json({"status": "completed", "result": res})
                return

            with _JOB_LOCK:
                if _JOB_STATE["running"]:
                    self._send_json({"error": "Job already running"}, status=409)
                    return
                _JOB_STATE["running"] = True
                _JOB_STATE["progress"] = 4
                _JOB_STATE["message"] = "Running ephemeral in-RAM evaluation on server..."
                _JOB_STATE["error"] = None
                _JOB_STATE["result"] = None

            t = threading.Thread(
                target=_background_in_memory_worker,
                args=(bundle, job_cfg, dataset_label),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started"})
            return

        self._send_json({"error": f"Unknown endpoint: {route}"}, status=404)


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    status = get_system_status()
    print("=" * 74)
    print("  FEDASSESS — FEDERATED VIABILITY INDEX (FVI) PLATFORM")
    print("=" * 74)
    print(f"  Host / Node      : {status['hostname']} (Env: {status['conda_env']})")
    print(f"  Web Framework    : {status['web_framework']}")
    print(f"  Compute Device   : {status['gpu_name']} ({status['backend']}, BF16: {status['bf16_supported']})")
    print("  Storage Policy   : ZERO DISK STORAGE (100% Browser or Ephemeral In-RAM)")
    print(f"  Web URL          : http://{host}:{port}")
    print("=" * 74)

    if FASTAPI_AVAILABLE:
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    else:
        server = ThreadingHTTPServer((host, port), FVIRequestHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down FVI Web Server...")
            server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the FedAssess FVI Web Platform Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (use 0.0.0.0 on rama for LAN access)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
