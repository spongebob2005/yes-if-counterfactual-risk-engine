"""Friction simulator (M5). All numbers are configurable assumptions (config.yaml `policy.sim`).

Public datasets record no interventions or their outcomes, so a step-up's catch rate or a
hold's abandonment rate cannot be measured here; they are stated, not learned.
"""

from __future__ import annotations

import numpy as np

ACTIONS = ("ALLOW", "STEP_UP", "HOLD", "BLOCK")


def realized_cost(action: str, is_fraud: bool, amount: float, rng: np.random.Generator, sim: dict, margin: float):
    """Sample one outcome. Returns (total, fraud_loss, friction, legit_loss)."""
    s = sim[action]
    friction, fraud_loss, legit_loss = float(s["fric"]), 0.0, 0.0
    if is_fraud:
        if rng.random() > s["catch"]:
            fraud_loss = amount  # fraud slips through: full loss
    elif rng.random() < s["abandon"]:
        legit_loss = margin * amount  # a legitimate customer walks away
    return friction + fraud_loss + legit_loss, fraud_loss, friction, legit_loss


def expected_cost(action: str, r, amount, sim: dict, margin: float):
    s = sim[action]
    return s["fric"] + r * (1 - s["catch"]) * amount + (1 - r) * s["abandon"] * margin * amount


def expected_cost_matrix(r, amount, sim: dict, margin: float) -> np.ndarray:
    """(n, 4) expected cost of every action, columns in ACTIONS order."""
    r, amount = np.asarray(r, dtype=float), np.asarray(amount, dtype=float)
    return np.column_stack([expected_cost(a, r, amount, sim, margin) for a in ACTIONS])


def drifting(sim: dict, drift_index: int, catch_after: float):
    """Non-stationary world: from `drift_index` on, attackers have learned to pass step-up."""
    after = {**sim, "STEP_UP": {**sim["STEP_UP"], "catch": catch_after}}

    def sim_at(i: int) -> dict:
        return sim if i < drift_index else after

    return sim_at
