import unittest

from risk_engine import (
    SAMPLE_TRANSACTIONS,
    action_for_score,
    counterfactual_options,
    policy_cost_series,
    score_transaction,
)


class RiskEngineTests(unittest.TestCase):
    def test_score_is_bounded_and_explains_positive_signals(self):
        transaction = SAMPLE_TRANSACTIONS[1]
        score, factors = score_transaction(transaction)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 0.98)
        self.assertTrue(factors)
        self.assertEqual(factors, sorted(factors, key=lambda item: item[1], reverse=True))

    def test_amount_and_delay_adjustments_are_applied(self):
        transaction = SAMPLE_TRANSACTIONS[1]
        baseline, _ = score_transaction(transaction)
        reduced_amount, _ = score_transaction(transaction, amount=100)
        longer_delay, _ = score_transaction(transaction, minutes_since_last=90)

        self.assertLess(reduced_amount, baseline)
        self.assertLess(longer_delay, baseline)

    def test_actions_follow_score_bands(self):
        self.assertEqual(action_for_score(0.10), "ALLOW")
        self.assertEqual(action_for_score(0.30), "STEP UP")
        self.assertEqual(action_for_score(0.60), "HOLD")
        self.assertEqual(action_for_score(0.80), "BLOCK")

    def test_counterfactual_options_are_ranked_by_reduction_per_friction(self):
        options = counterfactual_options(SAMPLE_TRANSACTIONS[1])
        ratios = [
            float(option["reduction"]) / float(option["friction"])
            for option in options
        ]

        self.assertEqual(ratios, sorted(ratios, reverse=True))
        self.assertEqual(len(options), 3)

    def test_policy_curves_have_drift_experiment_shape(self):
        curves = policy_cost_series(rounds=80, drift_round=45)

        self.assertEqual(set(curves), {
            "Allow most", "Static rules", "Step-up first", "Adaptive friction"
        })
        self.assertTrue(all(len(values) == 80 for values in curves.values()))
        self.assertTrue(all(values[-1] > values[0] for values in curves.values()))


if __name__ == "__main__":
    unittest.main()
