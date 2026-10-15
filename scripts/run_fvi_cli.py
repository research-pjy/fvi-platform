#!/usr/bin/env python3
"""
Headless Command-Line Interface (CLI) for the Federated Viability Index (`fvi-platform`).

Allows you to run full FVI benchmarks directly from the terminal on `rama` (L40S)
without opening a browser, saving the Markdown report, JSON metrics, and PNG
dashboard charts into `outputs/`.

Usage on `rama`:
    conda activate qwen-vl
    PYTHONPATH=src python scripts/run_fvi_cli.py --data-path data/pe_malware_features.csv --target malware_family --clients 10 --alpha 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from fvi.ingestion.sample_datasets import ensure_sample_datasets_exist
from fvi.pipeline import run_fvi_benchmark_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Federated Viability Index (FVI) Benchmark Study from the CLI."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/pe_malware_features.csv",
        help="Path to CSV file or ImageFolder directory (default: data/pe_malware_features.csv).",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="auto",
        choices=["auto", "tabular", "image"],
    )
    parser.add_argument(
        "--target",
        type=str,
        default="malware_family",
        help="Target column name (for CSV datasets).",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["all"],
        help="Feature column names (or 'all').",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of simulated federated clients M (default: 10).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="dirichlet",
        choices=["dirichlet", "iid", "quantity_skew", "generalist_specialist"],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet concentration parameter alpha (default: 0.1).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Global communication rounds for Iterative FL (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cli_run",
        help="Directory to save reports and diagnostic figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_sample_datasets_exist(ROOT_DIR / "data")

    job_cfg = FVJobConfig(
        dataset=DatasetConfig(
            data_path=args.data_path,
            modality=args.modality,
            target_column=args.target,
            feature_columns=args.features,
        ),
        partition=PartitionConfig(
            num_clients=args.clients,
            strategy=args.strategy,
            dirichlet_alpha=args.alpha,
        ),
        model=ModelConfig(arch="auto"),
        benchmark=BenchmarkConfig(fl_global_rounds=args.rounds),
        hardware=HardwareConfig(),
        output_dir=args.output_dir,
    )

    def _cli_progress(pct: int, msg: str) -> None:
        print(f"  [{pct:3d}%] {msg}")

    print("=" * 74)
    print("  RUNNING FEDERATED VIABILITY INDEX (FVI) CLI BENCHMARK")
    print("=" * 74)
    result = run_fvi_benchmark_pipeline(job_cfg, progress_cb=_cli_progress)
    print("\n" + result["markdown_report"])
    print(f"[Saved Markdown Report] -> {result['report_path']}")
    print(f"[Saved Visual Dashboard] -> {Path(args.output_dir) / 'fvi_benchmark_dashboard.png'}")


if __name__ == "__main__":
    main()
