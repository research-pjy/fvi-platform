"""
Classification Evaluation Metrics (Accuracy, Macro-F1, Weighted-F1, Per-Class Accuracy,
and Confusion Matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics on a validation or test split."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_accuracy: List[float]
    per_class_f1: List[float]
    confusion_matrix: List[List[int]]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> ClassificationMetrics:
    """
    Computes overall accuracy (%), Macro-F1 (%), Weighted-F1 (%), per-class
    accuracy (%), and confusion matrix in pure NumPy.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    n = max(1, len(yt))

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for r, c in zip(yt, yp):
        if 0 <= r < num_classes and 0 <= c < num_classes:
            conf[r, c] += 1

    acc = 100.0 * float(np.trace(conf)) / float(n)

    per_class_acc: List[float] = []
    per_class_f1: List[float] = []
    supports = conf.sum(axis=1)
    pred_totals = conf.sum(axis=0)

    for c in range(num_classes):
        tp = float(conf[c, c])
        sup = float(supports[c])
        p_tot = float(pred_totals[c])

        c_acc = (100.0 * tp / sup) if sup > 0 else 0.0
        prec = (tp / p_tot) if p_tot > 0 else 0.0
        rec = (tp / sup) if sup > 0 else 0.0
        f1 = (200.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_acc.append(c_acc)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    total_sup = float(supports.sum())
    if total_sup > 0:
        weighted_f1 = float(np.sum(np.array(per_class_f1) * supports) / total_sup)
    else:
        weighted_f1 = macro_f1

    return ClassificationMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_accuracy=per_class_acc,
        per_class_f1=per_class_f1,
        confusion_matrix=conf.tolist(),
    )
