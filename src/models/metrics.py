"""Detector metrics (M2): PR-AUC, ROC-AUC, recall at 1% FPR and calibration error."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y, p, target: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target, fpr, tpr))


def ece(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += abs(y[mask].mean() - p[mask].mean()) * mask.mean()
    return float(total)


def reliability(y, p, bins: int = 10) -> list[dict[str, float]]:
    """Mean predicted vs observed fraud rate per probability bin (for the reliability plot)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() >= 20:
            out.append({"predicted": float(p[mask].mean()), "observed": float(y[mask].mean()), "n": int(mask.sum())})
    return out


def report(y, p) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_1pct_fpr": recall_at_fpr(y, p),
        "ece": ece(y, p),
    }
