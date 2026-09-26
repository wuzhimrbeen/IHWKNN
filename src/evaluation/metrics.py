"""Evaluation metrics shared by IHWKNN and baseline experiments."""

from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.metrics import average_precision_score, precision_recall_curve


EPS = 1e-12


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = y_score >= threshold
    y_true_bool = y_true.astype(bool)

    tp = float(np.sum(y_pred & y_true_bool))
    fp = float(np.sum(y_pred & ~y_true_bool))
    tn = float(np.sum(~y_pred & ~y_true_bool))
    fn = float(np.sum(~y_pred & y_true_bool))

    tpr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)
    precision = tp / (tp + fp + EPS)
    f1 = 2.0 * precision * tpr / (precision + tpr + EPS)

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
    }


def max_f1_from_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_values = 2.0 * precision * recall / (precision + recall + EPS)
    best_idx = int(np.nanargmax(f1_values))

    if best_idx >= len(thresholds):
        best_threshold = float(np.max(y_score))
    else:
        best_threshold = float(thresholds[best_idx])

    return {
        "F1_max": float(f1_values[best_idx]),
        "Precision_at_F1_max": float(precision[best_idx]),
        "Recall_at_F1_max": float(recall[best_idx]),
        "best_threshold": best_threshold,
    }


def topk_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: Optional[int] = None,
) -> dict[str, float]:
    positives = int(np.sum(y_true == 1))
    if k is None:
        k = positives
    k = max(1, min(int(k), len(y_true)))

    order = np.argsort(y_score)[::-1]
    top_idx = order[:k]
    hits = float(np.sum(y_true[top_idx] == 1))

    precision_at_k = hits / k
    recall_at_k = hits / (positives + EPS)
    f1_at_k = 2.0 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k + EPS)

    return {
        "Precision_at_K": precision_at_k,
        "Recall_at_K": recall_at_k,
        "F1_at_K": f1_at_k,
        "K": float(k),
    }


def ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    metrics = binary_metrics_at_threshold(y_true, y_score, threshold)
    metrics["AUPR"] = float(average_precision_score(y_true, y_score))
    metrics.update(max_f1_from_pr_curve(y_true, y_score))
    metrics.update(topk_metrics(y_true, y_score, top_k))
    return metrics
