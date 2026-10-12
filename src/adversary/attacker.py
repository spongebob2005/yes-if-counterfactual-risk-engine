"""Black-box attacker (M6).

Threat model: the attacker can query the decision a limited number of times per attempt,
controls only the same actionable fields as the counterfactual engine (amount, timing,
funding source) and keeps at least half the amount so the fraud stays worth doing.
Success (evasion) means the final risk is below tau_allow. Perturbations happen in feature
space, which is a simplification: real attackers face stricter constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

CARD6_OPTS = ("debit", "credit")
RiskFn = Callable[[Sequence[Mapping[str, Any]]], np.ndarray]


def mutate(x: Mapping[str, Any], rng: np.random.Generator, min_amt_frac: float = 0.5) -> dict[str, Any]:
    y = dict(x)
    y["TransactionAmt"] = round(float(x["TransactionAmt"]) * rng.uniform(min_amt_frac, 1.0), 2)
    y["TransactionDT"] = int(x["TransactionDT"]) + int(rng.integers(0, 24)) * 3600
    y["card6"] = CARD6_OPTS[int(rng.integers(2))]
    return y


def random_search(x, risk_fn: RiskFn, rng, budget: int, tau: float, min_amt_frac: float = 0.5):
    """Coarse-output attacker: only learns whether a query was allowed, so it can only sample."""
    cands = [mutate(x, rng, min_amt_frac) for _ in range(budget)]
    risks = risk_fn(cands)
    i = int(np.argmin(risks))
    return cands[i], float(risks[i]), bool((risks < tau).any())


def hill_climb(x, risk_fn: RiskFn, rng, budget: int, tau: float, min_amt_frac: float = 0.5):
    """Open-score attacker: samples like the coarse attacker with half its queries, then uses the
    exact scores it saw to search around the best candidate with the rest."""
    first = max(1, budget // 2)
    best, best_r, ok = random_search(x, risk_fn, rng, first, tau, min_amt_frac)
    used, floor = first, float(x["TransactionAmt"]) * min_amt_frac
    while used < budget and not ok:
        batch = []
        for _ in range(min(5, budget - used)):
            y = dict(best)
            y["TransactionAmt"] = round(max(floor, float(best["TransactionAmt"]) * rng.uniform(0.85, 1.05)), 2)
            shift = int(best["TransactionDT"]) + int(rng.integers(-3, 4)) * 3600
            y["TransactionDT"] = min(max(shift, int(x["TransactionDT"])), int(x["TransactionDT"]) + 23 * 3600)
            if rng.random() < 0.2:
                y["card6"] = CARD6_OPTS[int(rng.integers(2))]
            batch.append(y)
        risks = risk_fn(batch)
        used += len(batch)
        i = int(np.argmin(risks))
        if risks[i] < best_r:
            best, best_r = batch[i], float(risks[i])
        ok = best_r < tau
    return best, best_r, ok


def evasion_rate(rows: Sequence[Mapping[str, Any]], risk_fn: RiskFn, rng, budget: int, tau: float,
                 strategy=random_search, min_amt_frac: float = 0.5) -> float:
    if not rows:
        return 0.0
    wins = sum(strategy(x, risk_fn, rng, budget, tau, min_amt_frac)[2] for x in rows)
    return wins / len(rows)
