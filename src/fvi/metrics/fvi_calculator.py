"""
Federated Viability Index (FVI) Calculator & Scientific Verdict Synthesizer.

Mathematical Formulation:
  Given:
    - A_cent:  Centralized Upper-Bound Test Accuracy (%) on 100% pooled data
    - A_local: Sample-weighted average Isolated Silo Test Accuracy (%) (Lower Bound)
    - A_FL:    Federated Learning Test Accuracy (%) under the target partition
    - A_IID:   Federated Learning Test Accuracy (%) under an IID reference split

  Sub-metrics (each in [0, 1]):
    1. Centralized Parity Ratio (CPR):
         CPR = clip(A_FL / max(A_cent, 1e-6), 0.0, 1.0)
    2. Collaboration Gain (CG):
         CG = clip((A_FL - A_local) / max(A_cent - A_local, 1.0), 0.0, 1.0)
    3. Heterogeneity Resilience (HR):
         HR = clip(A_FL / max(A_IID, 1e-6), 0.0, 1.0)

  Composite Federated Viability Index (FVI in [0, 100]):
         FVI = 100 * (0.45 * CPR + 0.35 * CG + 0.20 * HR)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class AlgorithmFVIEntry:
    """FVI breakdown for a single Federated Learning algorithm."""

    algorithm_name: str
    test_accuracy: float
    test_macro_f1: float
    comm_mib: float
    cpr: float
    cg: float
    hr: float
    fvi_score: float
    delta_vs_centralized: float
    gain_over_isolated: float


@dataclass
class FVIScorecard:
    """Complete Federated Viability Index Scorecard for the dataset and partition."""

    overall_fvi_score: float
    verdict_tier: str        # "High Viability", "Moderate Viability", or "Low Viability"
    verdict_color: str       # "emerald", "amber", or "rose"
    best_fl_algorithm: str
    best_fl_accuracy: float
    best_fl_macro_f1: float
    centralized_accuracy: float
    centralized_macro_f1: float
    isolated_weighted_accuracy: float
    isolated_weighted_macro_f1: float
    iid_reference_accuracy: float
    cpr_score: float         # in [0, 100]
    cg_score: float          # in [0, 100]
    hr_score: float          # in [0, 100]
    executive_summary: str
    recommendation: str
    per_algorithm_scores: List[AlgorithmFVIEntry]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_single_fvi(
    a_fl: float,
    a_cent: float,
    a_local: float,
    a_iid: float,
) -> Tuple[float, float, float, float]:
    """
    Returns `(cpr, cg, hr, fvi_score)` where `cpr, cg, hr` are in `[0, 1]`
    and `fvi_score` is in `[0, 100]`.
    """
    cpr = float(np.clip(a_fl / max(1e-6, a_cent), 0.0, 1.0))
    gap_denom = max(2.0, a_cent - a_local)
    cg = float(np.clip((a_fl - a_local) / gap_denom, 0.0, 1.0))
    hr = float(np.clip(a_fl / max(1e-6, a_iid), 0.0, 1.0))
    fvi = 100.0 * (0.45 * cpr + 0.35 * cg + 0.20 * hr)
    return cpr, cg, hr, float(np.clip(fvi, 0.0, 100.0))


def compute_fvi_scorecard(
    centralized_acc: float,
    centralized_f1: float,
    isolated_acc: float,
    isolated_f1: float,
    iid_ref_acc: float,
    fl_results: List[Dict[str, Any]],
    partition_description: str,
) -> FVIScorecard:
    """
    Synthesizes the `FVIScorecard` across all evaluated FL algorithms.
    """
    entries: List[AlgorithmFVIEntry] = []

    for r in fl_results:
        a_fl = float(r["test_accuracy"])
        f1_fl = float(r["test_macro_f1"])
        comm_mib = float(r["comm_mib"])
        cpr, cg, hr, fvi = compute_single_fvi(
            a_fl=a_fl,
            a_cent=centralized_acc,
            a_local=isolated_acc,
            a_iid=iid_ref_acc,
        )
        entries.append(
            AlgorithmFVIEntry(
                algorithm_name=str(r["name"]),
                test_accuracy=round(a_fl, 2),
                test_macro_f1=round(f1_fl, 2),
                comm_mib=round(comm_mib, 3),
                cpr=round(cpr * 100.0, 1),
                cg=round(cg * 100.0, 1),
                hr=round(hr * 100.0, 1),
                fvi_score=round(fvi, 1),
                delta_vs_centralized=round(a_fl - centralized_acc, 2),
                gain_over_isolated=round(a_fl - isolated_acc, 2),
            )
        )

    entries.sort(key=lambda e: e.fvi_score, reverse=True)
    best = entries[0]

    if best.fvi_score >= 80.0:
        tier = "High Viability"
        color = "emerald"
        summary = (
            f"Federated Learning is highly viable for this dataset under {partition_description}. "
            f"{best.algorithm_name} achieves {best.test_accuracy:.2f}% test accuracy "
            f"(retaining {best.cpr:.1f}% of the {centralized_acc:.2f}% centralized upper bound) "
            f"and boosts accuracy by {best.gain_over_isolated:+.2f}% over isolated client silos."
        )
        rec = (
            f"Deploy {best.algorithm_name} as your primary federated strategy. "
            "Collaborative training provides strong accuracy gains without sharing raw data."
        )
    elif best.fvi_score >= 60.0:
        tier = "Moderate Viability"
        color = "amber"
        summary = (
            f"Federated Learning shows moderate viability under {partition_description}. "
            f"While statistical heterogeneity introduces a {abs(best.delta_vs_centralized):.2f}% gap "
            f"relative to centralized pooling ({centralized_acc:.2f}%), {best.algorithm_name} "
            f"still outperforms isolated client training ({isolated_acc:.2f}%) by {best.gain_over_isolated:+.2f}%."
        )
        rec = (
            f"Prefer heterogeneity-resilient methods such as {best.algorithm_name} (FENS or FedProx) "
            "over naive FedAvg to mitigate non-IID client drift."
        )
    else:
        tier = "Low Viability"
        color = "rose"
        summary = (
            f"Federated Learning exhibits low viability under {partition_description} "
            f"(FVI = {best.fvi_score:.1f}/100). Extreme statistical skew or limited per-client "
            f"samples restrict {best.algorithm_name} to {best.test_accuracy:.2f}% vs. {centralized_acc:.2f}% centralized."
        )
        rec = (
            "Consider feature alignment, two-phase logit ensembling (FENS), or increasing local "
            "client sample diversity before transitioning to live multi-party federation."
        )

    return FVIScorecard(
        overall_fvi_score=best.fvi_score,
        verdict_tier=tier,
        verdict_color=color,
        best_fl_algorithm=best.algorithm_name,
        best_fl_accuracy=best.test_accuracy,
        best_fl_macro_f1=best.test_macro_f1,
        centralized_accuracy=round(centralized_acc, 2),
        centralized_macro_f1=round(centralized_f1, 2),
        isolated_weighted_accuracy=round(isolated_acc, 2),
        isolated_weighted_macro_f1=round(isolated_f1, 2),
        iid_reference_accuracy=round(iid_ref_acc, 2),
        cpr_score=best.cpr,
        cg_score=best.cg,
        hr_score=best.hr,
        executive_summary=summary,
        recommendation=rec,
        per_algorithm_scores=entries,
    )
