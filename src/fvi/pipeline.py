"""
Master Orchestration Pipeline (`FVIPipeline`) for the Federated Viability Index Platform.

Coordinates:
  1. Dataset loading & preprocessing (`fvi.ingestion`)
  2. Simulated client partitioning & entropy diagnostics (`fvi.partition`)
  3. Mode 1 unified model instantiation (`fvi.models`)
  4. Execution of the Golden Triad:
     - Centralized Upper Bound
     - Isolated Local Silos (Lower Bound)
     - Optional IID Reference Run (for Heterogeneity Resilience `HR`)
     - Selected Federated Learning Strategies (`FedAvg`, `FedProx`, `FedAdam`, `FENS`)
  5. Synthesis of the 0-100 Federated Viability Index (`FVIScorecard`), diagnostic
     charts, and downloadable Markdown & JSON reports.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from fvi.config import FVJobConfig, PartitionConfig
from fvi.engines.benchmark_runner import (
    AlgorithmRunResult,
    run_centralized_upper_bound,
    run_fens_fl,
    run_isolated_local_silos,
    run_iterative_fl,
)
from fvi.ingestion.loader import load_and_preprocess_dataset
from fvi.metrics.fvi_calculator import compute_fvi_scorecard
from fvi.metrics.plots import (
    plot_benchmark_dashboard_base64,
    plot_partition_preview_base64,
)
from fvi.models.factory import TORCH_AVAILABLE, build_unified_model_spec
from fvi.partition.splitter import partition_dataset


ProgressCallback = Callable[[int, str], None]


def preview_partition_only(job_cfg: FVJobConfig) -> Dict[str, Any]:
    """
    Fast (<1 second) partition preview called by the Web UI when the user clicks
    "Preview Client Partition" before launching training.
    """
    bundle = load_and_preprocess_dataset(job_cfg.dataset)
    partition = partition_dataset(
        bundle.y_train, job_cfg.partition, num_classes=bundle.num_classes
    )
    model_spec = build_unified_model_spec(bundle, job_cfg.model)
    chart_b64 = plot_partition_preview_base64(
        class_counts=partition.class_counts_full,
        client_entropies=partition.client_normalized_entropy,
        class_names=bundle.class_names,
        description=partition.description,
    )
    return {
        "partition": partition.to_summary_dict(bundle.class_names),
        "model_spec": {
            "arch_name": model_spec.arch_name,
            "modality": model_spec.modality,
            "input_dim": model_spec.input_dim,
            "num_classes": model_spec.num_classes,
            "num_parameters": model_spec.num_parameters,
            "fp32_size_mib": round(model_spec.fp32_size_mib, 3),
            "backend": "PyTorch (CUDA/CPU)" if TORCH_AVAILABLE else "NumPy Engine",
        },
        "partition_chart_base64": chart_b64,
    }


def _generate_markdown_report(
    job_cfg: FVJobConfig,
    partition_summary: Dict[str, Any],
    model_spec_dict: Dict[str, Any],
    scorecard_dict: Dict[str, Any],
    isolated_dict: Dict[str, Any],
    fl_results_dicts: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Federated Viability Index (FVI) Benchmark Report",
        "",
        f"- **Dataset Source:** `{job_cfg.dataset.data_path}` (`{model_spec_dict['modality']}`)",
        f"- **Target Variable:** `{job_cfg.dataset.target_column}` ({model_spec_dict['num_classes']} classes: `{', '.join(partition_summary['class_names'])}`)",
        f"- **Client Partitioning:** {partition_summary['description']} (Mean Client Entropy = `{partition_summary['mean_normalized_entropy']}`)",
        f"- **Mode 1 Unified Client Model:** `{model_spec_dict['arch_name']}` (`{model_spec_dict['num_parameters']:,}` parameters, `{model_spec_dict['fp32_size_mib']:.3f} MiB` in FP32)",
        "",
        "## 1. Executive Scorecard: Federated Viability Index (FVI)",
        "",
        f"- **Composite FVI Score:** **{scorecard_dict['overall_fvi_score']:.1f} / 100** (**{scorecard_dict['verdict_tier']}**)",
        f"- **Top Federated Strategy:** **{scorecard_dict['best_fl_algorithm']}** (`{scorecard_dict['best_fl_accuracy']:.2f}%` Test Accuracy | `{scorecard_dict['best_fl_macro_f1']:.2f}%` Macro-F1)",
        f"- **Centralized Parity Ratio (CPR):** `{scorecard_dict['cpr_score']:.1f}%` (Centralized Upper Bound = `{scorecard_dict['centralized_accuracy']:.2f}%`)",
        f"- **Collaboration Gain (CG):** `{scorecard_dict['cg_score']:.1f}%` (Isolated Silos Weighted Mean = `{scorecard_dict['isolated_weighted_accuracy']:.2f}%`)",
        f"- **Heterogeneity Resilience (HR):** `{scorecard_dict['hr_score']:.1f}%` (IID Reference = `{scorecard_dict['iid_reference_accuracy']:.2f}%`)",
        "",
        f"> **Assessment:** {scorecard_dict['executive_summary']}",
        f"> **Recommendation:** {scorecard_dict['recommendation']}",
        "",
        "## 2. Golden Triad & Federated Algorithm Comparison",
        "",
        "| Benchmark / Strategy | Test Accuracy (%) | Macro-F1 (%) | Delta vs. Centralized | Gain vs. Isolated Silos | Client Comm. (MiB) | FVI Score |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Centralized (Pooled Upper Bound)** | **{scorecard_dict['centralized_accuracy']:.2f}%** | {scorecard_dict['centralized_macro_f1']:.2f}% | 0.00% | {scorecard_dict['centralized_accuracy'] - scorecard_dict['isolated_weighted_accuracy']:+.2f}% | 0.000 | — |",
        f"| **Isolated Client Silos (Lower Bound)** | **{scorecard_dict['isolated_weighted_accuracy']:.2f}%** | {scorecard_dict['isolated_weighted_macro_f1']:.2f}% | {scorecard_dict['isolated_weighted_accuracy'] - scorecard_dict['centralized_accuracy']:+.2f}% | 0.00% | 0.000 | — |",
    ]

    for entry in scorecard_dict["per_algorithm_scores"]:
        lines.append(
            f"| **{entry['algorithm_name']}** | **{entry['test_accuracy']:.2f}%** | {entry['test_macro_f1']:.2f}% | "
            f"{entry['delta_vs_centralized']:+.2f}% | {entry['gain_over_isolated']:+.2f}% | "
            f"{entry['comm_mib']:.3f} | **{entry['fvi_score']:.1f}** |"
        )

    lines.extend(
        [
            "",
            "## 3. Isolated Client Silo Breakdown",
            "",
            "| Client ID | Local Training Samples | Normalized Label Entropy [0–1] | Isolated Test Accuracy (%) | Isolated Macro-F1 (%) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for cid in range(partition_summary["num_clients"]):
        lines.append(
            f"| **Client {cid:02d}** | {isolated_dict['per_client_samples'][cid]} | "
            f"{partition_summary['client_normalized_entropy'][cid]:.3f} | "
            f"{isolated_dict['per_client_accuracy'][cid]:.2f}% | "
            f"{isolated_dict['per_client_macro_f1'][cid]:.2f}% |"
        )

    return "\n".join(lines) + "\n"


def run_fvi_benchmark_pipeline(
    job_cfg: FVJobConfig,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Executes the complete Federated Viability Index benchmark study and returns
    a JSON-serializable dictionary containing all metrics, base64 plots, and
    saved report paths.
    """
    def _notify(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    t_total = time.perf_counter()
    out_dir = Path(job_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _notify(5, f"Loading and preprocessing dataset: {job_cfg.dataset.data_path}...")
    bundle = load_and_preprocess_dataset(job_cfg.dataset)

    _notify(12, f"Partitioning training split across M={job_cfg.partition.num_clients} clients ({job_cfg.partition.strategy})...")
    partition = partition_dataset(
        bundle.y_train, job_cfg.partition, num_classes=bundle.num_classes
    )
    model_spec = build_unified_model_spec(bundle, job_cfg.model)

    part_chart_b64 = plot_partition_preview_base64(
        class_counts=partition.class_counts_full,
        client_entropies=partition.client_normalized_entropy,
        class_names=bundle.class_names,
        description=partition.description,
        save_path=out_dir / "partition_distribution.png",
    )

    _notify(20, f"Training Centralized Pooled Upper-Bound Model ({model_spec.arch_name})...")
    centralized_res = run_centralized_upper_bound(
        bundle, job_cfg.model, job_cfg.benchmark, job_cfg.hardware
    )

    _notify(35, f"Training {partition.num_clients} Isolated Client Silos (Lower-Bound Baseline)...")
    isolated_res = run_isolated_local_silos(
        bundle, partition, job_cfg.model, job_cfg.benchmark, job_cfg.hardware
    )

    fl_results: List[AlgorithmRunResult] = []
    req_algos = [a.lower() for a in job_cfg.benchmark.algorithms]

    iter_algos = [a for a in req_algos if a in ("fedavg", "fedprox", "fedadam")]
    for idx, algo in enumerate(iter_algos):
        pct = 48 + int(25 * (idx / max(1, len(iter_algos))))
        _notify(pct, f"Running Iterative FL Strategy: {algo.upper()} ({job_cfg.benchmark.fl_global_rounds} rounds)...")
        res = run_iterative_fl(
            algorithm=algo,
            bundle=bundle,
            partition=partition,
            model_spec=model_spec,
            model_cfg=job_cfg.model,
            bench_cfg=job_cfg.benchmark,
            hw_cfg=job_cfg.hardware,
        )
        fl_results.append(res)

    if "fens" in req_algos or "fedens" in req_algos:
        _notify(78, "Running FENS (Phase 1 One-Shot + INT8 Quantization + Phase 2 FedAdam Aggregator)...")
        fedens_res, fens_res = run_fens_fl(
            bundle=bundle,
            partition=partition,
            model_spec=model_spec,
            model_cfg=job_cfg.model,
            bench_cfg=job_cfg.benchmark,
            hw_cfg=job_cfg.hardware,
        )
        fl_results.append(fedens_res)
        fl_results.append(fens_res)

    if not fl_results:
        fl_results.append(
            run_iterative_fl(
                "fedavg",
                bundle,
                partition,
                model_spec,
                job_cfg.model,
                job_cfg.benchmark,
                job_cfg.hardware,
            )
        )

    # Compute IID reference accuracy for Heterogeneity Resilience (HR)
    if partition.strategy == "iid" or not job_cfg.benchmark.run_iid_reference_for_hr:
        iid_ref_acc = max(r.test_metrics.accuracy for r in fl_results)
    else:
        _notify(90, "Computing IID Reference Split for Heterogeneity Resilience (HR)...")
        iid_part_cfg = copy.deepcopy(job_cfg.partition)
        iid_part_cfg.strategy = "iid"
        iid_part = partition_dataset(
            bundle.y_train, iid_part_cfg, num_classes=bundle.num_classes
        )
        iid_res = run_iterative_fl(
            "fedavg",
            bundle,
            iid_part,
            model_spec,
            job_cfg.model,
            job_cfg.benchmark,
            job_cfg.hardware,
        )
        iid_ref_acc = max(
            iid_res.test_metrics.accuracy,
            max(r.test_metrics.accuracy for r in fl_results),
        )

    _notify(95, "Synthesizing Federated Viability Index (FVI) Scorecard & Charts...")
    fl_dicts = [r.to_summary_dict() for r in fl_results]
    scorecard = compute_fvi_scorecard(
        centralized_acc=centralized_res.test_metrics.accuracy,
        centralized_f1=centralized_res.test_metrics.macro_f1,
        isolated_acc=isolated_res.weighted_test_accuracy,
        isolated_f1=isolated_res.weighted_test_macro_f1,
        iid_ref_acc=iid_ref_acc,
        fl_results=fl_dicts,
        partition_description=partition.description,
    )
    scorecard_dict = scorecard.to_dict()
    isolated_dict = isolated_res.to_summary_dict()
    partition_summary = partition.to_summary_dict(bundle.class_names)
    model_spec_dict = {
        "arch_name": model_spec.arch_name,
        "modality": model_spec.modality,
        "input_dim": model_spec.input_dim,
        "num_classes": model_spec.num_classes,
        "num_parameters": model_spec.num_parameters,
        "fp32_size_mib": round(model_spec.fp32_size_mib, 3),
        "backend": "PyTorch (CUDA/CPU)" if TORCH_AVAILABLE else "NumPy Engine",
    }

    dashboard_b64 = plot_benchmark_dashboard_base64(
        scorecard_dict=scorecard_dict,
        isolated_dict=isolated_dict,
        fl_results_list=fl_dicts,
        save_path=out_dir / "fvi_benchmark_dashboard.png",
    )

    md_report = _generate_markdown_report(
        job_cfg=job_cfg,
        partition_summary=partition_summary,
        model_spec_dict=model_spec_dict,
        scorecard_dict=scorecard_dict,
        isolated_dict=isolated_dict,
        fl_results_dicts=fl_dicts,
    )
    md_path = out_dir / "fvi_benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    full_payload = {
        "total_wall_time_sec": round(time.perf_counter() - t_total, 2),
        "config": job_cfg.to_dict(),
        "model_spec": model_spec_dict,
        "partition": partition_summary,
        "centralized": centralized_res.to_summary_dict(),
        "isolated": isolated_dict,
        "fl_results": fl_dicts,
        "scorecard": scorecard_dict,
        "partition_chart_base64": part_chart_b64,
        "dashboard_chart_base64": dashboard_b64,
        "markdown_report": md_report,
        "report_path": str(md_path),
    }

    with open(out_dir / "fvi_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in full_payload.items() if not k.endswith("_base64")},
            f,
            indent=2,
        )

    _notify(100, f"Benchmark Complete! Overall FVI = {scorecard.overall_fvi_score:.1f}/100 ({scorecard.verdict_tier})")
    return full_payload
