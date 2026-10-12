"""Baseline policies (B1-B3) and the stream replay used for the drift experiment (E5)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .bandit import LinUCB, ThompsonFriction, context
from .simulator import ACTIONS, expected_cost_matrix, realized_cost


def single_threshold(r: float, t: dict) -> int:
    """B1: block above tau_block, else allow."""
    return 3 if r >= t["tau_block"] else 0


def static_ladder(r: float, t: dict) -> int:
    """B2: fixed bands allow / step-up / hold / block."""
    if r < t["tau_allow"]:
        return 0
    if r < t["tau_hold"]:
        return 1
    if r < t["tau_block"]:
        return 2
    return 3


def analytic(sim: dict, margin: float) -> Callable[[float, float], int]:
    """B3: minimise expected cost under a fixed belief about the simulator."""

    def choose(r: float, amount: float) -> int:
        return int(np.argmin(expected_cost_matrix([r], [amount], sim, margin)[0]))

    return choose


def replay(stream: dict[str, np.ndarray], choose: Callable[[int], int], sim_at, margin: float,
           seed: int, update: Callable[[int, int, dict], None] | None = None) -> dict[str, np.ndarray]:
    """Replay a time-ordered stream; the policy only sees the outcome of the action it took."""
    rng = np.random.default_rng(seed)
    n = len(stream["risk"])
    parts = np.zeros((n, 3))
    actions = np.zeros(n, dtype=int)
    for i in range(n):
        a = choose(i)
        total, fraud, friction, legit = realized_cost(
            ACTIONS[a], bool(stream["y"][i]), float(stream["amount"][i]), rng, sim_at(i), margin)
        parts[i] = (fraud, friction, legit)
        actions[i] = a
        if update is not None:
            update(i, a, {"cost": total, "fraud": bool(stream["y"][i]), "slipped": fraud > 0, "abandoned": legit > 0})
    return {"cumulative": np.cumsum(parts.sum(axis=1)), "parts": parts.sum(axis=0), "actions": actions}


def run_bandit(stream, sim_at, margin: float, alpha: float, clip: float, seed: int,
               bandit: LinUCB | None = None, gamma: float = 1.0, tau_allow: float = 0.0):
    """Below tau_allow the payment is allowed; inside the review band the bandit picks the friction
    (step-up, hold or block), exactly as the deployed engine does."""
    bandit = bandit or LinUCB(len(ACTIONS), 4, alpha, gamma)
    contexts = [context(stream["risk"][i], stream["amount"][i]) for i in range(len(stream["risk"]))]
    in_band = stream["risk"] >= tau_allow
    friction_only = np.array([False, True, True, True])

    def choose(i: int) -> int:
        return bandit.choose(contexts[i], allowed=friction_only) if in_band[i] else 0

    def update(i: int, a: int, outcome: dict) -> None:
        if in_band[i]:
            bandit.update(a, contexts[i], -min(outcome["cost"], clip) / 100.0)  # clipped, scaled reward

    result = replay(stream, choose, sim_at, margin, seed, update)
    return result, bandit


def run_thompson(stream, sim_at, margin: float, seed: int, policy: ThompsonFriction, tau_allow: float):
    """Same gate as the bandit: allow below tau_allow, otherwise pick the cheapest friction."""
    rng = np.random.default_rng(seed + 7)
    in_band = stream["risk"] >= tau_allow
    friction_only = np.array([False, True, True, True])

    def choose(i: int) -> int:
        if not in_band[i]:
            return 0
        return policy.choose(float(stream["risk"][i]), float(stream["amount"][i]), friction_only, rng)

    def update(i: int, a: int, outcome: dict) -> None:
        if in_band[i]:
            policy.update(a, outcome["fraud"], outcome["slipped"], outcome["abandoned"])

    return replay(stream, choose, sim_at, margin, seed, update), policy
