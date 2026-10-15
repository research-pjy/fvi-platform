"""
Evaluation Metrics, Federated Viability Index (FVI) Calculator, and Diagnostic Plots.
"""

from fvi.metrics.evaluator import ClassificationMetrics, compute_classification_metrics
from fvi.metrics.fvi_calculator import FVIScorecard, compute_fvi_scorecard
from fvi.metrics.plots import (
    plot_benchmark_dashboard_base64,
    plot_partition_preview_base64,
)

__all__ = [
    "ClassificationMetrics",
    "compute_classification_metrics",
    "FVIScorecard",
    "compute_fvi_scorecard",
    "plot_partition_preview_base64",
    "plot_benchmark_dashboard_base64",
]
