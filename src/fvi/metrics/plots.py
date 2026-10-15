"""
Minimalist, Publication-Grade Matplotlib Diagnostic Plots for `fvi-platform`.

Plots FVI, CPR, CG, and HR on the [0.00, 1.00] dimensionless index scale while
keeping classification accuracies on the [0, 100]% scale.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_base64_and_file(fig: plt.Figure, save_path: Optional[Path] = None) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(buf.getvalue())
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def plot_partition_preview_base64(
    class_counts: np.ndarray,
    client_entropies: Sequence[float],
    class_names: Sequence[str],
    description: str,
    save_path: Optional[Path] = None,
) -> str:
    """
    Renders a 2-panel client partition diagnostic plot:
      - Left: Bubble matrix of client (x) vs. class (y) sample counts.
      - Right: Per-client normalized Shannon label entropy H_i in [0, 1].
    """
    num_clients, num_classes = class_counts.shape
    fig, (ax_bub, ax_ent) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    max_cnt = max(1.0, float(class_counts.max()))
    cmap = plt.get_cmap("tab10")

    for cid in range(num_clients):
        for c in range(num_classes):
            cnt = class_counts[cid, c]
            if cnt > 0:
                sz = 260.0 * (cnt / max_cnt)
                ax_bub.scatter(
                    cid,
                    c,
                    s=max(10.0, sz),
                    color=cmap(c % 10),
                    alpha=0.82,
                    edgecolors="#1E293B",
                    linewidth=0.5,
                )

    ax_bub.set_xticks(range(num_clients))
    ax_bub.set_xticklabels([f"C{i}" for i in range(num_clients)], fontsize=8.5)
    ax_bub.set_yticks(range(num_classes))
    ax_bub.set_yticklabels([str(name)[:14] for name in class_names], fontsize=8.5)
    ax_bub.set_xlabel("Simulated Client ID", fontsize=9.5, fontweight="bold")
    ax_bub.set_ylabel("Target Class", fontsize=9.5, fontweight="bold")
    ax_bub.set_title("(a) Client-Class Sample Distribution", fontsize=10.5, fontweight="bold")
    ax_bub.grid(True, linestyle=":", alpha=0.45)

    x_c = np.arange(num_clients)
    bars = ax_ent.bar(
        x_c,
        client_entropies,
        color="#1D4ED8",
        alpha=0.85,
        edgecolor="#1E293B",
        linewidth=0.6,
    )
    ax_ent.axhline(
        float(np.mean(client_entropies)),
        color="#D97706",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Entropy = {np.mean(client_entropies):.2f}",
    )
    for b, val in zip(bars, client_entropies):
        ax_ent.text(
            b.get_x() + b.get_width() / 2.0,
            val + 0.03,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax_ent.set_xticks(x_c)
    ax_ent.set_xticklabels([f"C{i}" for i in range(num_clients)], fontsize=8.5)
    ax_ent.set_ylim(0, 1.15)
    ax_ent.set_xlabel("Simulated Client ID", fontsize=9.5, fontweight="bold")
    ax_ent.set_ylabel("Normalized Shannon Entropy [0.00–1.00]", fontsize=9.5, fontweight="bold")
    ax_ent.set_title("(b) Client Label Diversity Index (1.00 = Balanced)", fontsize=10.5, fontweight="bold")
    ax_ent.legend(loc="lower right", fontsize=8.5)
    ax_ent.grid(axis="y", linestyle=":", alpha=0.45)

    fig.suptitle(description, fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _fig_to_base64_and_file(fig, save_path=save_path)


def plot_benchmark_dashboard_base64(
    scorecard_dict: Dict[str, Any],
    isolated_dict: Dict[str, Any],
    fl_results_list: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> str:
    """
    Renders a 4-panel visual summary of the Federated Viability Index benchmark,
    with Panel (d) plotting FVI, CPR, and CG on the [0.00, 1.00] unit index scale.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax_triad, ax_conv = axes[0, 0], axes[0, 1]
    ax_clients, ax_fvi = axes[1, 0], axes[1, 1]

    # Panel (a): Golden Triad Accuracy & Macro-F1 (%)
    names = ["Centralized\n(Upper Bound)", "Isolated Silos\n(Lower Bound)"] + [
        r["name"].replace(" (Iterative)", "").replace(" (Two-Phase Ensembling)", "\n(Ours)")
        for r in fl_results_list
    ]
    accs = [
        scorecard_dict["centralized_accuracy"],
        scorecard_dict["isolated_weighted_accuracy"],
    ] + [r["test_accuracy"] for r in fl_results_list]
    f1s = [
        scorecard_dict["centralized_macro_f1"],
        scorecard_dict["isolated_weighted_macro_f1"],
    ] + [r["test_macro_f1"] for r in fl_results_list]

    x = np.arange(len(names))
    w = 0.36
    b1 = ax_triad.bar(
        x - w / 2, accs, width=w, label="Test Accuracy (%)", color="#111827", edgecolor="#0F172A"
    )
    b2 = ax_triad.bar(
        x + w / 2, f1s, width=w, label="Macro-F1 (%)", color="#2563EB", edgecolor="#0F172A"
    )
    for b, val in zip(b1, accs):
        ax_triad.text(
            b.get_x() + b.get_width() / 2.0,
            val + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    ax_triad.set_xticks(x)
    ax_triad.set_xticklabels(names, fontsize=8.5, rotation=12)
    ax_triad.set_ylim(0, 110)
    ax_triad.set_ylabel("Percentage [%]", fontsize=9.5, fontweight="bold")
    ax_triad.set_title("(a) Golden Triad Benchmark: Accuracy & Macro-F1", fontsize=10.5, fontweight="bold")
    ax_triad.legend(fontsize=8.5, loc="lower right")
    ax_triad.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): Convergence Curves
    ax_conv.axhline(
        scorecard_dict["centralized_accuracy"],
        color="#059669",
        linestyle="--",
        linewidth=1.5,
        label="Centralized Upper Bound",
    )
    ax_conv.axhline(
        scorecard_dict["isolated_weighted_accuracy"],
        color="#BE123C",
        linestyle=":",
        linewidth=1.5,
        label="Isolated Silos Mean",
    )
    palette = ["#1D4ED8", "#7C3AED", "#D97706", "#059669", "#DC2626"]
    for idx, r in enumerate(fl_results_list):
        hist = r.get("round_history", [])
        if len(hist) > 1:
            rnds = [h["round"] for h in hist]
            t_accs = [h["test_acc"] for h in hist]
            ax_conv.plot(
                rnds,
                t_accs,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=palette[idx % len(palette)],
                label=r["name"],
            )
        elif len(hist) == 1:
            ax_conv.scatter(
                [1],
                [hist[0]["test_acc"]],
                s=65,
                color=palette[idx % len(palette)],
                label=r["name"],
                zorder=5,
            )

    ax_conv.set_xlabel("Communication Round / Step", fontsize=9.5, fontweight="bold")
    ax_conv.set_ylabel("Global Test Accuracy [%]", fontsize=9.5, fontweight="bold")
    ax_conv.set_ylim(0, 105)
    ax_conv.set_title("(b) Federated Convergence Trajectory", fontsize=10.5, fontweight="bold")
    ax_conv.legend(fontsize=8, loc="lower right")
    ax_conv.grid(True, linestyle=":", alpha=0.45)

    # Panel (c): Per-Client Isolated Silo Accuracy vs Best FL Accuracy
    c_accs = isolated_dict["per_client_accuracy"]
    c_x = np.arange(len(c_accs))
    ax_clients.bar(
        c_x,
        c_accs,
        color="#94A3B8",
        edgecolor="#1E293B",
        label="Client Solo Accuracy (Isolated Silo)",
    )
    ax_clients.axhline(
        scorecard_dict["best_fl_accuracy"],
        color="#1D4ED8",
        linewidth=2.2,
        linestyle="-",
        label=f"Best FL ({scorecard_dict['best_fl_algorithm']}: {scorecard_dict['best_fl_accuracy']:.1f}%)",
    )
    ax_clients.set_xticks(c_x)
    ax_clients.set_xticklabels([f"C{i}" for i in c_x], fontsize=8.5)
    ax_clients.set_ylim(0, 105)
    ax_clients.set_xlabel("Client ID", fontsize=9.5, fontweight="bold")
    ax_clients.set_ylabel("Test Accuracy [%]", fontsize=9.5, fontweight="bold")
    ax_clients.set_title("(c) Client Collaboration Gain Over Solo Training", fontsize=10.5, fontweight="bold")
    ax_clients.legend(fontsize=8.5, loc="lower right")
    ax_clients.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (d): FVI Scorecard Breakdown in [0.00, 1.00]
    entries = scorecard_dict["per_algorithm_scores"]
    e_names = [
        e["algorithm_name"].replace(" (Iterative)", "").replace(" (Two-Phase Ensembling)", "")
        for e in entries
    ]
    fvi_vals = [e["fvi_score"] for e in entries]
    cpr_vals = [e["cpr"] for e in entries]
    cg_vals = [e["cg"] for e in entries]

    ex = np.arange(len(e_names))
    bw = 0.25
    ax_fvi.bar(ex - bw, fvi_vals, width=bw, label="Composite FVI [0–1]", color="#111827")
    ax_fvi.bar(ex, cpr_vals, width=bw, label="Centralized Parity (CPR)", color="#2563EB")
    ax_fvi.bar(ex + bw, cg_vals, width=bw, label="Collaboration Gain (CG)", color="#059669")

    for idx, val in enumerate(fvi_vals):
        ax_fvi.text(
            ex[idx] - bw,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax_fvi.set_xticks(ex)
    ax_fvi.set_xticklabels(e_names, fontsize=8.5, rotation=12)
    ax_fvi.set_ylim(0, 1.15)
    ax_fvi.set_ylabel("Index Value [0.00–1.00]", fontsize=9.5, fontweight="bold")
    ax_fvi.set_title("(d) Federated Viability Index (FVI) Breakdown [0–1]", fontsize=10.5, fontweight="bold")
    ax_fvi.legend(fontsize=8, loc="lower right")
    ax_fvi.grid(axis="y", linestyle=":", alpha=0.45)

    fig.suptitle(
        f"Federated Viability Index Report — Overall FVI: {scorecard_dict['overall_fvi_score']:.3f} ({scorecard_dict['verdict_tier']})",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _fig_to_base64_and_file(fig, save_path=save_path)
