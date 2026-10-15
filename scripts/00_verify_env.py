#!/usr/bin/env python3
from __future__ import annotations
import importlib, os, platform, shutil, socket, sys, tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from fvi.config import BenchmarkConfig, DatasetConfig, FVJobConfig, ModelConfig, PartitionConfig
from fvi.ingestion.sample_datasets import generate_pe_malware_tabular_csv
from fvi.pipeline import run_fvi_benchmark_pipeline

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("Pillow", "PIL"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]

def main() -> None:
    hostname = socket.gethostname()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV") or "(system)"
    disk_free_gib = shutil.disk_usage(ROOT_DIR).free / (1024.0 ** 3)
    print("=" * 74)
    print("  FEDASSESS (FVI PLATFORM) — ENVIRONMENT & HARDWARE VERIFICATION")
    print("=" * 74)
    print(f"  Hostname         : {hostname}")
    print(f"  Active Env       : {conda_env}")
    print(f"  Python Version   : {platform.python_version()}")
    print(f"  Free Disk Space  : {disk_free_gib:.1f} GiB")
    print("-" * 74)
    for display_name, mod_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    [OK] {display_name:14s} : {ver}")
        except ImportError:
            print(f"    [--] {display_name:14s} : NOT INSTALLED (run `pip install -r requirements.txt`)")
    print("-" * 74)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA Available   : {cuda_ok} (CUDA runtime: {torch.version.cuda})")
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            bf16_ok = torch.cuda.is_bf16_supported()
            print(f"  GPU Device       : {gpu_name}")
            print(f"  GPU VRAM         : {free_b / (1024**3):.2f} GiB free / {total_b / (1024**3):.2f} GiB total")
            print(f"  BF16 Supported   : {bf16_ok}")
    except ImportError:
        print("  CUDA / GPU Check : PyTorch not installed yet (using NumPy fallback engine)")
    print("-" * 74)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "smoke_malware.csv"
        out_dir = Path(tmp) / "smoke_out"
        generate_pe_malware_tabular_csv(csv_path, num_samples=250, seed=42)
        cfg = FVJobConfig(
            dataset=DatasetConfig(data_path=str(csv_path), modality="tabular", target_column="malware_family"),
            partition=PartitionConfig(num_clients=4, strategy="dirichlet", dirichlet_alpha=0.1),
            model=ModelConfig(arch="auto", hidden_dims=[32, 16]),
            benchmark=BenchmarkConfig(
                algorithms=["fedavg", "fens"],
                centralized_epochs=3,
                isolated_epochs=3,
                fl_global_rounds=3,
                fl_local_epochs=1,
                fens_phase1_epochs=3,
                fens_phase2_rounds=10,
                run_iid_reference_for_hr=False,
            ),
            output_dir=str(out_dir),
        )
        res = run_fvi_benchmark_pipeline(cfg)
        print(f"    [OK] Smoke Test Succeeded! Sample FVI Score = {res["scorecard"]["overall_fvi_score"]:.1f}/100")
    print("=" * 74)

if __name__ == "__main__":
    main()
