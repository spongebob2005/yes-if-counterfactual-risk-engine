"""yes, if: the decision layer on top of a calibrated fraud detector.

`RiskEngine` loads the artifacts written by `python -m src.experiments.run_all` (models/)
and answers one question per payment: allow, step up, hold or block, and if not allow,
what is the cheapest change that would make the model say yes. Numpy only at runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.counterfactual.search import counterfactuals, describe
from src.data.core import FEATURES, LABELS, feature_matrix, rows_to_columns
from src.models.serving import RiskModel
from src.policy.bandit import LinUCB, ThompsonFriction, context
from src.policy.simulator import ACTIONS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

GUIDANCE = {
    "ALLOW": "Approved.",
    "STEP_UP": "Additional verification is needed to complete this payment.",
    "HOLD": "This payment is being reviewed. You'll hear back shortly.",
    "BLOCK": "This payment couldn't be approved.",
}

# Raw transaction fields; anything missing is filled from training-set defaults.
RAW_FIELDS = (
    "TransactionAmt", "TransactionDT", "ProductCD", "card4", "card6", "dist1", "P_emaildomain",
    "DeviceType", "prior_cnt", "prior_mean_amt", "prev_dt", "card_device_prior",
    "DeviceInfo_prior_cnt", "DeviceInfo_distinct_uids", "P_emaildomain_prior_cnt",
    "P_emaildomain_distinct_uids", "addr1_prior_cnt", "addr1_distinct_uids",
)


class RiskEngine:
    def __init__(self, model: RiskModel, meta: dict[str, Any]):
        self.model = model
        self.meta = meta
        self.features = tuple(meta["features"])
        self.levels = meta["levels"]
        self.thresholds = meta["thresholds"]
        policy = meta["policy"]
        self.bandit = LinUCB.from_state(policy["bandit"]) if policy.get("bandit") else None
        self.learned = ThompsonFriction.from_state(policy["thompson"]) if policy.get("thompson") else None
        self.policy_mode = policy["mode"]

    @classmethod
    def load(cls, directory: str | Path = MODELS_DIR) -> "RiskEngine":
        directory = Path(directory)
        meta = json.loads((directory / "engine.json").read_text(encoding="utf-8"))
        return cls(RiskModel.load(directory / "detector.npz"), meta)

    # -- inputs ------------------------------------------------------------
    def complete(self, tx: Mapping[str, Any]) -> dict[str, Any]:
        """Fill missing history/graph fields: per email domain if known, else training medians."""
        out = {key: tx.get(key) for key in RAW_FIELDS}
        domain = out.get("P_emaildomain") or self.meta["defaults"]["P_emaildomain"]
        out["P_emaildomain"] = domain
        per_domain = self.meta["email_stats"].get(str(domain), self.meta["email_stats"]["__unknown__"])
        for key, value in per_domain.items():
            if out.get(key) is None:
                out[key] = value
        for key, value in self.meta["defaults"].items():
            if out.get(key) is None and key != "prev_dt":
                out[key] = value
        return out

    # -- scoring -----------------------------------------------------------
    def matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return feature_matrix(rows_to_columns(rows), self.levels, self.features)

    def score(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return self.model.predict(self.matrix(rows))

    def explain(self, tx: Mapping[str, Any], k: int = 5) -> list[dict[str, Any]]:
        """Top reasons, as path contributions to the model's log-odds grouped by plain label."""
        contrib, _ = self.model.trees.contributions(self.matrix([tx]), len(self.features))
        grouped: dict[str, float] = {}
        for name, value in zip(self.features, contrib[0]):
            label = str(LABELS.get(name, name))
            grouped[label] = grouped.get(label, 0.0) + float(value)
        top = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
        return [{"reason": label, "impact": round(value, 3), "direction": "raises" if value > 0 else "lowers"}
                for label, value in top]

    def choose_action(self, risk: float, tx: Mapping[str, Any]) -> str:
        """Thresholds bound the decision; inside the review band the learned policy picks the friction."""
        t = self.thresholds
        if risk < t["tau_allow"]:
            return "ALLOW"
        if risk >= t["tau_block"]:
            return "BLOCK"
        friction_only = np.array([False, True, True, True])
        if self.policy_mode == "thompson" and self.learned is not None:
            return ACTIONS[self.learned.choose(risk, float(tx["TransactionAmt"]), friction_only)]
        if self.policy_mode == "bandit" and self.bandit is not None:
            x = context(risk, float(tx["TransactionAmt"]))
            return ACTIONS[self.bandit.choose(x, allowed=friction_only, alpha=0.0)]
        return "STEP_UP" if risk < t["tau_hold"] else "HOLD"

    def counterfactuals(self, tx: Mapping[str, Any], k: int | None = None) -> list[dict[str, Any]]:
        cfg = self.meta["counterfactual"]
        return counterfactuals(tx, self.score, self.thresholds["tau_allow"], cfg, k or cfg.get("top_k", 3))

    def decide(self, tx: Mapping[str, Any], view: str = "user") -> dict[str, Any]:
        """Analysts see the score, reasons and exact counterfactuals; users see coarse guidance only."""
        full = self.complete(tx)
        risk = float(self.score([full])[0])
        action = self.choose_action(risk, full)
        if view != "analyst":
            return {"action": action, "guidance": GUIDANCE[action]}
        options = [] if action == "ALLOW" else self.counterfactuals(full)
        return {
            "action": action,
            "guidance": GUIDANCE[action],
            "risk": round(risk, 4),
            "thresholds": self.thresholds,
            "top_reasons": self.explain(full),
            "counterfactuals": [
                {"label": "yes, if " + describe(o["change"], full["card6"]),
                 "change": o["change"], "kind": list(o["kind"]), "risk": round(o["risk"], 4),
                 "cost": round(o["cost"], 3), "plausible": o["plausible"],
                 "amount": o["transaction"]["TransactionAmt"]}
                for o in options
            ],
            "transaction": full,
        }


@lru_cache(maxsize=1)
def default_engine() -> RiskEngine:
    return RiskEngine.load()


__all__ = ["RiskEngine", "default_engine", "GUIDANCE", "FEATURES", "MODELS_DIR"]
