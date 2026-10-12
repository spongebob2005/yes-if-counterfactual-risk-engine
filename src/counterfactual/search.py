"""Counterfactual engine (M4): the smallest actionable change that brings risk below tau_allow.

Only three things are actionable; everything else (device, history, graph signals) is
immutable. Counterfactuals describe the *model's* decision boundary, not causal truth:
"the model would accept this if...".

| Change              | Meaning                                  | Range               |
| TransactionAmt      | pay part now, the rest after verification | 10% to 100%         |
| TransactionDT       | schedule the payment for later           | 0 to 24 hours later |
| card6               | pay with a different funding source      | debit or credit     |
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULTS = {
    "amt_mults": (1.0, 0.9, 0.75, 0.5, 0.25, 0.1),
    "delays_h": (0, 1, 2, 4, 8, 12, 24),
    "card6_opts": ("debit", "credit"),
    "weights": {"amt": 1.0, "delay": 0.5, "method": 0.3},
}


def change_cost(x: Mapping[str, Any], y: Mapping[str, Any], w: Mapping[str, float]) -> float:
    amt = abs(np.log(y["TransactionAmt"] / x["TransactionAmt"]))
    delay_days = (y["TransactionDT"] - x["TransactionDT"]) / 86400
    method = float(y["card6"] != x["card6"])
    return float(w["amt"] * amt + w["delay"] * delay_days + w["method"] * method)


def candidates(x: Mapping[str, Any], cfg: Mapping[str, Any] = DEFAULTS):
    """Every grid point except the original (6 amounts x 7 delays x 2 methods - 1 = 83)."""
    for m, d, c in itertools.product(cfg["amt_mults"], cfg["delays_h"], cfg["card6_opts"]):
        if m == 1.0 and d == 0 and c == x["card6"]:
            continue
        y = dict(x)
        y["TransactionAmt"] = round(float(x["TransactionAmt"]) * m, 2)
        y["TransactionDT"] = int(x["TransactionDT"]) + int(d * 3600)
        y["card6"] = c
        yield y, {"amount_mult": m, "delay_h": d, "card6": c}


def kind(change: Mapping[str, Any], original_card6: str) -> tuple[str, ...]:
    parts = []
    if change["amount_mult"] != 1.0:
        parts.append("amount")
    if change["delay_h"]:
        parts.append("delay")
    if change["card6"] != original_card6:
        parts.append("method")
    return tuple(parts)


def describe(change: Mapping[str, Any], original_card6: str) -> str:
    """Plain words for a change, e.g. "half is paid now and it goes through 8 hours later"."""
    parts = []
    mult = float(change["amount_mult"])
    if mult < 1:
        parts.append(f"{mult:.0%} is paid now and the rest after verification")
    if change["delay_h"]:
        hours = int(change["delay_h"])
        parts.append(f"it goes through {hours} hour{'s' if hours != 1 else ''} later")
    if change["card6"] != original_card6:
        parts.append(f"it is paid by {change['card6']} card")
    return " and ".join(parts) or "nothing changes"


def plausible(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
    """Inside the card's own history: no more than 3x its usual spend (new cards always pass)."""
    if not x.get("prior_cnt"):
        return True
    return y["TransactionAmt"] <= 3 * float(x["prior_mean_amt"])


def counterfactuals(
    x: Mapping[str, Any],
    risk_fn: Callable[[Sequence[Mapping[str, Any]]], np.ndarray],
    tau: float,
    cfg: Mapping[str, Any] = DEFAULTS,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Score all candidates in one batch; keep only minimal valid changes (no smaller change also
    works), then return up to k of the cheapest, diverse by kind."""
    x_card = x["card6"]
    pairs = list(candidates(x, cfg))
    risks = risk_fn([y for y, _ in pairs])
    valid = []
    for (y, change), r in zip(pairs, risks):
        if r < tau:
            valid.append({
                "cost": change_cost(x, y, cfg["weights"]), "risk": float(r), "change": change,
                "kind": kind(change, x["card6"]), "plausible": plausible(x, y), "transaction": y,
            })
    valid = [y for y in valid if not any(_smaller(x["change"], y["change"], x_card6=x["transaction"]["card6"],
                                                   y_card6=y["transaction"]["card6"], original=x_card)
                                          for x in valid if x is not y)]
    valid.sort(key=lambda item: (item["cost"], item["risk"]))
    chosen, seen = [], set()
    for item in valid:  # one per kind first (amount-only, delay-only, method-only, ...)
        if item["kind"] not in seen:
            chosen.append(item)
            seen.add(item["kind"])
        if len(chosen) == k:
            break
    for item in valid:
        if len(chosen) == k:
            break
        if item not in chosen:
            chosen.append(item)
    chosen.sort(key=lambda item: (item["cost"], item["risk"]))
    return chosen


def _smaller(a: Mapping[str, Any], b: Mapping[str, Any], *, x_card6: str, y_card6: str, original: str) -> bool:
    """True if change `a` is no bigger than `b` on every axis and differs: `b` is then not minimal."""
    no_bigger = (a["amount_mult"] >= b["amount_mult"] and a["delay_h"] <= b["delay_h"]
                 and (x_card6 == original or x_card6 == y_card6))
    return no_bigger and a != b


def quality(results: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    """E4: validity, proximity, sparsity and plausibility of the best counterfactual per row."""
    best = [r[0] for r in results if r]
    return {
        "flagged": len(results),
        "validity": len(best) / max(len(results), 1),
        "proximity": float(np.mean([b["cost"] for b in best])) if best else float("nan"),
        "sparsity": float(np.mean([len(b["kind"]) for b in best])) if best else float("nan"),
        "plausibility": float(np.mean([b["plausible"] for b in best])) if best else float("nan"),
    }
