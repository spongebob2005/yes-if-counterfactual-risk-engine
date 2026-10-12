"""Detector and calibration (M2): LightGBM + isotonic, a logistic baseline, cost-based thresholds."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.policy.simulator import ACTIONS, expected_cost_matrix

from .serving import RiskModel, TreeEnsemble


def fit_detector(Xtr, ytr, Xva, yva, Xcal, ycal, params: dict[str, Any], seed: int = 42, sample_weight=None,
                 monotone: list[int] | None = None):
    """Early stopping on val (PR-AUC), isotonic calibration on a separate calib slice.

    `monotone` (one of -1/0/+1 per feature) keeps the model sensible on the fields a
    counterfactual can change: paying less never raises risk.
    """
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        subsample_freq=1,
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        random_state=seed,
        verbose=-1,
        monotone_constraints=monotone,
        monotone_constraints_method="advanced" if monotone else "basic",
    )
    model.fit(
        Xtr, ytr, sample_weight=sample_weight, eval_X=(Xva,), eval_y=(yva,), eval_metric="average_precision",
        callbacks=[lgb.early_stopping(params["early_stopping"], verbose=False)],
    )
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.predict_proba(Xcal)[:, 1], ycal)
    return model, iso


def export(model, iso) -> RiskModel:
    """Flatten to the numpy predictor and verify it matches LightGBM exactly."""
    risk = RiskModel(TreeEnsemble.from_lightgbm(model.booster_), iso.X_thresholds_, iso.y_thresholds_)
    return risk


def check_parity(model, iso, risk: RiskModel, X: np.ndarray) -> float:
    expected = iso.predict(model.predict_proba(X)[:, 1])
    worst = float(np.max(np.abs(expected - risk.predict(X))))
    if worst > 1e-9:
        raise AssertionError(f"numpy predictor differs from LightGBM by {worst}")
    return worst


def fit_logreg(Xtr, ytr, seed: int = 42):
    """E1 baseline: scaled logistic regression on the same features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None, random_state=seed))
    model.fit(Xtr, ytr)
    return model


def choose_thresholds(risk: np.ndarray, y: np.ndarray, amount: np.ndarray, sim: dict, margin: float) -> dict[str, float]:
    """Pick the allow / step-up / hold / block ladder that minimises expected cost on validation.

    Each row's cost under each action uses its true label (r = y), so this is the cost the
    static ladder would actually have paid on the validation stream.
    """
    order = np.argsort(risk)
    r, cost = risk[order], expected_cost_matrix(y[order].astype(float), amount[order], sim, margin)
    cum = np.vstack([np.zeros(len(ACTIONS)), np.cumsum(cost, axis=0)])
    grid = np.unique(np.quantile(r, np.linspace(0.5, 0.999, 60)))
    pos = np.searchsorted(r, grid)
    best = (np.inf, None)
    for i, a in enumerate(pos):
        for j in range(i, len(pos)):
            b = pos[j]
            for k in range(j, len(pos)):
                c = pos[k]
                total = (cum[a, 0] + (cum[b, 1] - cum[a, 1]) + (cum[c, 2] - cum[b, 2]) + (cum[-1, 3] - cum[c, 3]))
                if total < best[0]:
                    best = (total, (grid[i], grid[j], grid[k]))
    tau_allow, tau_hold, tau_block = (float(t) for t in best[1])
    return {"tau_allow": tau_allow, "tau_hold": tau_hold, "tau_block": tau_block}
