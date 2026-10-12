import json
import unittest

import numpy as np

from risk_engine import GUIDANCE, MODELS_DIR, RiskEngine


class RiskEngineTests(unittest.TestCase):
    """Runs against the committed artifacts in models/ (regenerate with src.experiments.run_all)."""

    @classmethod
    def setUpClass(cls):
        cls.engine = RiskEngine.load()
        cls.samples = {s["key"]: s for s in json.loads((MODELS_DIR / "samples.json").read_text())}

    def decide(self, key, view="analyst"):
        return self.engine.decide(self.samples[key]["transaction"], view)

    def test_clean_purchase_is_allowed_and_known_fraud_is_blocked(self):
        self.assertEqual(self.decide("clean")["action"], "ALLOW")
        for key in ("ring", "suspicious", "testing"):
            self.assertIn(self.decide(key)["action"], ("HOLD", "BLOCK"), key)

    def test_user_view_hides_scores_and_counterfactuals(self):
        user = self.decide("borderline", "user")
        self.assertEqual(set(user), {"action", "guidance"})
        self.assertEqual(user["guidance"], GUIDANCE[user["action"]])

    def test_counterfactuals_bring_risk_under_the_allow_line(self):
        result = self.decide("large")
        self.assertNotEqual(result["action"], "ALLOW")
        self.assertTrue(result["counterfactuals"])
        for option in result["counterfactuals"]:
            self.assertLess(option["risk"], self.engine.thresholds["tau_allow"])
            self.assertTrue(option["label"].startswith("yes, if "))

    def test_paying_less_never_raises_risk(self):
        tx = self.engine.complete(self.samples["large"]["transaction"])
        rows = [dict(tx, TransactionAmt=tx["TransactionAmt"] * m) for m in (1.0, 0.75, 0.5, 0.25, 0.1)]
        risks = self.engine.score(rows)
        self.assertTrue(np.all(np.diff(risks) <= 1e-12), risks)

    def test_missing_history_is_filled_from_training_defaults(self):
        full = self.engine.complete({"TransactionAmt": 50.0, "TransactionDT": 15 * 3600, "card6": "debit"})
        self.assertIsNone(full["prev_dt"])
        self.assertIsNotNone(full["prior_mean_amt"])
        self.assertIn(self.engine.decide(full)["action"], GUIDANCE)

    def test_reasons_are_plain_language(self):
        reasons = self.decide("ring")["top_reasons"]
        self.assertEqual(reasons[0]["reason"], "Cards sharing this device")
        self.assertEqual(reasons[0]["direction"], "raises")


if __name__ == "__main__":
    unittest.main()
