#!/usr/bin/env python3
"""
FastAPI Web Server & REST API for FedAssess (`fvi-platform`).

Supports:
  1. True Drag-and-Drop Binary Uploads (`POST /api/upload-dataset`):
     - Accepts `.csv` files (tabular datasets or image manifests) and `.zip`
       archives (containing class subfolders of images for ImageFolder tasks)
     - Streams uploaded files directly into `data/uploads/` on `rama`
     - Automatically extracts `.zip` image archives and locates the class-folder root
  2. Native FastAPI + Uvicorn execution (`python app/server.py --port 8080`)
     with automatic standard-library `ThreadingHTTPServer` fallback.
"""

from __future__ import annotations

import argparse
import cgi
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
import threading
import time
import traceback
from typing import Any, Dict
from urllib.parse import urlparse
import zipfile

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
from fvi.ingestion.schema import IMAGE_EXTENSIONS, inspect_dataset_source
from fvi.models.factory import TORCH_AVAILABLE
from fvi.pipeline import preview_partition_only, run_fvi_benchmark_pipeline

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

_JOB_LOCK = threading.Lock()
_JOB_STATE: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "message": "Idle",
    "error": None,
    "result": None,
}


def get_system_status() -> Dict[str, Any]:
    """Returns hardware, environment, and disk space metadata for the UI header."""
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

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    disk_free_gib = round(shutil.disk_usage(UPLOADS_DIR).free / (1024.0 ** 3), 1)

    return {
        "hostname": hostname,
        "conda_env": conda_env,
        "python_version": platform.python_version(),
        "web_framework": "FastAPI" if FASTAPI_AVAILABLE else "HTTP",
        "backend": "PyTorch" if TORCH_AVAILABLE else "NumPy",
        "gpu_name": gpu_name,
        "vram_gib": vram_gib,
        "bf16_supported": bf16_ok,
        "disk_free_gib": disk_free_gib,
        "uploads_dir": str(UPLOADS_DIR),
    }


def _sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return clean or f"dataset_{int(time.time())}.csv"


def _find_image_folder_root(extract_dir: Path) -> Path:
    """
    Walks an extracted `.zip` directory to find the folder whose subdirectories
    are class folders containing images (ignoring `__MACOSX` or single wrapper folders).
    """
    candidates = [extract_dir] + [
        p for p in extract_dir.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts
    ]
    for cand in candidates:
        subdirs = [
            d
            for d in cand.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "__MACOSX"
        ]
        if not subdirs:
            continue
        has_images = any(
            any(
                f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                for f in sd.iterdir()
            )
            for sd in subdirs
        )
        if has_images:
            return cand
    return extract_dir


def save_uploaded_dataset_bytes(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """
    Saves an uploaded `.csv` file or extracts an uploaded `.zip` archive into
    `data/uploads/` on `rama`, then inspects and returns its schema report.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename)
    suffix = Path(safe_name).suffix.lower()

    if suffix == ".zip":
        stem = Path(safe_name).stem
        extract_target = UPLOADS_DIR / stem
        if extract_target.exists():
            shutil.rmtree(extract_target)
        extract_target.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            zf.extractall(extract_target)

        img_root = _find_image_folder_root(extract_target)
        rel_path = str(img_root.relative_to(ROOT_DIR))
        report = inspect_dataset_source(img_root)
        rep_dict = report.to_dict()
        rep_dict["data_path"] = rel_path
        rep_dict["uploaded_filename"] = safe_name
        rep_dict["uploaded_size_mb"] = round(len(raw_bytes) / (1024.0 ** 2), 2)
        return rep_dict

    if suffix == ".csv":
        saved_path = UPLOADS_DIR / safe_name
        saved_path.write_bytes(raw_bytes)
        rel_path = f"data/uploads/{safe_name}"
        report = inspect_dataset_source(saved_path)
        rep_dict = report.to_dict()
        rep_dict["data_path"] = rel_path
        rep_dict["uploaded_filename"] = safe_name
        rep_dict["uploaded_size_mb"] = round(len(raw_bytes) / (1024.0 ** 2), 2)
        return rep_dict

    raise ValueError(
        f"Unsupported file format '{suffix}'. Please drop a .csv file (tabular) or a .zip archive of class folders (images)."
    )


def _parse_job_config(payload: Dict[str, Any]) -> FVJobConfig:
    ds_raw = payload.get("dataset", {})
    part_raw = payload.get("partition", {})
    mod_raw = payload.get("model", {})
    bench_raw = payload.get("benchmark", {})

    raw_path = str(ds_raw.get("data_path", "")).strip()
    if not raw_path:
        raise ValueError("No dataset selected. Please drop a dataset file first.")

    resolved_path = Path(raw_path)
    if not resolved_path.is_absolute() and not resolved_path.exists():
        resolved_path = ROOT_DIR / raw_path

    ds_cfg = DatasetConfig(
        data_path=str(resolved_path),
        modality=str(ds_raw.get("modality", "auto")),
        target_column=str(ds_raw.get("target_column", "")),
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
        output_dir=str(ROOT_DIR / "outputs" / "web_latest"),
    )


def _background_benchmark_worker(job_cfg: FVJobConfig) -> None:
    def _cb(pct: int, msg: str) -> None:
        with _JOB_LOCK:
            _JOB_STATE["progress"] = pct
            _JOB_STATE["message"] = msg

    try:
        res = run_fvi_benchmark_pipeline(job_cfg, progress_cb=_cb)
        with _JOB_LOCK:
            _JOB_STATE["running"] = False
            _JOB_STATE["progress"] = 100
            _JOB_STATE["message"] = "Completed"
            _JOB_STATE["result"] = res
            _JOB_STATE["error"] = None
    except Exception as exc:
        traceback.print_exc()
        with _JOB_LOCK:
            _JOB_STATE["running"] = False
            _JOB_STATE["error"] = str(exc)
            _JOB_STATE["message"] = f"Error: {exc}"


if FASTAPI_AVAILABLE:
    fastapi_app = FastAPI(
        title="FedAssess — Federated Viability Index (FVI)",
        version="0.3.0",
    )

    @fastapi_app.on_event("startup")
    def _on_startup() -> None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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

    @fastapi_app.post("/api/upload-dataset")
    async def api_upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
        try:
            raw_bytes = await file.read()
            if not raw_bytes:
                return JSONResponse(
                    content={"error": "Uploaded file is empty."}, status_code=400
                )
            rep_dict = save_uploaded_dataset_bytes(
                file.filename or "dataset.csv", raw_bytes
            )
            return JSONResponse(content=rep_dict)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    @fastapi_app.post("/api/upload-csv")
    async def api_upload_csv_json(request: Request) -> JSONResponse:
        payload = await request.json()
        filename = str(payload.get("filename", "dataset.csv"))
        content = str(payload.get("content", ""))
        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
        rep_dict = save_uploaded_dataset_bytes(filename, content.encode("utf-8"))
        return JSONResponse(content=rep_dict)

    @fastapi_app.post("/api/inspect-dataset")
    async def api_inspect_dataset(request: Request) -> JSONResponse:
        payload = await request.json()
        raw_path = str(payload.get("data_path", "")).strip()
        target_col = payload.get("target_column")
        p = Path(raw_path)
        if not p.is_absolute() and not p.exists():
            p = ROOT_DIR / raw_path
        report = inspect_dataset_source(p, target_column=target_col)
        report_dict = report.to_dict()
        report_dict["data_path"] = raw_path
        return JSONResponse(content=report_dict)

    @fastapi_app.post("/api/preview-partition")
    async def api_preview_partition(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            job_cfg = _parse_job_config(payload)
            preview_data = preview_partition_only(job_cfg)
            return JSONResponse(content=preview_data)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    @fastapi_app.post("/api/run-benchmark")
    async def api_run_benchmark(request: Request) -> JSONResponse:
        payload = await request.json()
        synchronous = bool(payload.get("synchronous", False))
        try:
            job_cfg = _parse_job_config(payload)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

        if synchronous:
            res = run_fvi_benchmark_pipeline(job_cfg)
            return JSONResponse(content={"status": "completed", "result": res})

        with _JOB_LOCK:
            if _JOB_STATE["running"]:
                return JSONResponse(
                    content={"error": "A benchmark is currently in progress."},
                    status_code=409,
                )
            _JOB_STATE["running"] = True
            _JOB_STATE["progress"] = 2
            _JOB_STATE["message"] = "Preparing Federated Viability evaluation..."
            _JOB_STATE["error"] = None
            _JOB_STATE["result"] = None

        t = threading.Thread(
            target=_background_benchmark_worker, args=(job_cfg,), daemon=True
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

        if route == "/api/upload-dataset":
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" in ctype:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": ctype,
                    },
                )
                if "file" not in form:
                    self._send_json({"error": "Missing file field."}, status=400)
                    return
                file_item = form["file"]
                filename = file_item.filename or "dataset.csv"
                raw_bytes = file_item.file.read()
                try:
                    rep = save_uploaded_dataset_bytes(filename, raw_bytes)
                    self._send_json(rep)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

        try:
            payload = self._read_json_body()
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return

        if route == "/api/upload-csv":
            try:
                filename = str(payload.get("filename", "dataset.csv"))
                content = str(payload.get("content", ""))
                rep = save_uploaded_dataset_bytes(filename, content.encode("utf-8"))
                self._send_json(rep)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if route == "/api/inspect-dataset":
            raw_path = str(payload.get("data_path", "")).strip()
            target_col = payload.get("target_column")
            p = Path(raw_path)
            if not p.is_absolute() and not p.exists():
                p = ROOT_DIR / raw_path
            report = inspect_dataset_source(p, target_column=target_col)
            rep_dict = report.to_dict()
            rep_dict["data_path"] = raw_path
            self._send_json(rep_dict)
            return

        if route == "/api/preview-partition":
            try:
                job_cfg = _parse_job_config(payload)
                self._send_json(preview_partition_only(job_cfg))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if route == "/api/run-benchmark":
            synchronous = bool(payload.get("synchronous", False))
            try:
                job_cfg = _parse_job_config(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            if synchronous:
                res = run_fvi_benchmark_pipeline(job_cfg)
                self._send_json({"status": "completed", "result": res})
                return

            with _JOB_LOCK:
                if _JOB_STATE["running"]:
                    self._send_json({"error": "Job already running"}, status=409)
                    return
                _JOB_STATE["running"] = True
                _JOB_STATE["progress"] = 2
                _JOB_STATE["message"] = "Preparing Federated Viability evaluation..."
                _JOB_STATE["error"] = None
                _JOB_STATE["result"] = None

            t = threading.Thread(
                target=_background_benchmark_worker, args=(job_cfg,), daemon=True
            )
            t.start()
            self._send_json({"status": "started"})
            return

        self._send_json({"error": f"Unknown endpoint: {route}"}, status=404)


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    status = get_system_status()
    print("=" * 74)
    print("  FEDASSESS — FEDERATED VIABILITY INDEX (FVI) PLATFORM")
    print("=" * 74)
    print(f"  Host / Node      : {status['hostname']} (Env: {status['conda_env']})")
    print(f"  Web Framework    : {status['web_framework']}")
    print(f"  Compute Device   : {status['gpu_name']} ({status['backend']}, BF16: {status['bf16_supported']})")
    print(f"  Uploads Storage  : {status['uploads_dir']} ({status['disk_free_gib']} GiB free)")
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
