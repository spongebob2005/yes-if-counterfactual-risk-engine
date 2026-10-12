import unittest

import numpy as np

from src.adversary.attacker import hill_climb, mutate, random_search
from src.counterfactual.search import candidates, counterfactuals, describe
from src.policy.bandit import LinUCB, ThompsonFriction, context
from src.policy.baselines import replay, run_thompson, static_ladder
from src.policy.simulator import ACTIONS, drifting, expected_cost, realized_cost

SIM = {
    "ALLOW": {"fric": 0.0, "catch": 0.0, "abandon": 0.0},
    "STEP_UP": {"fric": 2.0, "catch": 0.85, "abandon": 0.03},
    "HOLD": {"fric": 6.0, "catch": 0.9, "abandon": 0.15},
    "BLOCK": {"fric": 10.0, "catch": 1.0, "abandon": 1.0},
}
TX = {"TransactionAmt": 200.0, "TransactionDT": 3 * 3600, "card6": "credit", "prior_cnt": 10, "prior_mean_amt": 80.0}


def toy_risk(rows):
    """Risky when the amount is large, at night, or on credit."""
    out = []
    for r in rows:
        hour = (r["TransactionDT"] // 3600) % 24
        out.append(0.002 * r["TransactionAmt"] + (0.3 if hour < 6 else 0) + (0.1 if r["card6"] == "credit" else 0))
    return np.array(out)


class CounterfactualTests(unittest.TestCase):
    def test_grid_has_83_candidates_and_never_the_original(self):
        cands = list(candidates(TX))
        self.assertEqual(len(cands), 83)
        self.assertNotIn((1.0, 0, "credit"), [(c["amount_mult"], c["delay_h"], c["card6"]) for _, c in cands])

    def test_results_are_valid_cheapest_first_and_diverse(self):
        results = counterfactuals(TX, toy_risk, tau=0.3)
        self.assertTrue(results)
        self.assertTrue(all(r["risk"] < 0.3 for r in results))
        self.assertEqual([r["cost"] for r in results], sorted(r["cost"] for r in results))
        valid_kinds = {r["kind"] for r in counterfactuals(TX, toy_risk, tau=0.3, k=100)}
        self.assertEqual(len({r["kind"] for r in results}), min(3, len(valid_kinds)))

    def test_nothing_returned_when_no_change_is_enough(self):
        self.assertEqual(counterfactuals(TX, lambda rows: np.ones(len(rows)), tau=0.1), [])

    def test_describe_reads_naturally(self):
        self.assertEqual(describe({"amount_mult": 0.5, "delay_h": 8, "card6": "debit"}, "credit"),
                         "50% is paid now and the rest after verification and it goes through 8 hours later "
                         "and it is paid by debit card")


class SimulatorTests(unittest.TestCase):
    def test_expected_cost_matches_the_mean_of_sampled_costs(self):
        rng = np.random.default_rng(0)
        for action in ACTIONS:
            draws = [realized_cost(action, rng.random() < 0.3, 100.0, rng, SIM, 0.1)[0] for _ in range(20000)]
            self.assertAlmostEqual(np.mean(draws), expected_cost(action, 0.3, 100.0, SIM, 0.1), delta=1.0)

    def test_drift_only_weakens_step_up_after_the_drift_point(self):
        sim_at = drifting(SIM, 10, 0.4)
        self.assertEqual(sim_at(9)["STEP_UP"]["catch"], 0.85)
        self.assertEqual(sim_at(10)["STEP_UP"]["catch"], 0.4)
        self.assertEqual(sim_at(10)["HOLD"], SIM["HOLD"])

    def test_static_ladder_bands(self):
        t = {"tau_allow": 0.1, "tau_hold": 0.4, "tau_block": 0.8}
        self.assertEqual([static_ladder(r, t) for r in (0.05, 0.2, 0.5, 0.9)], [0, 1, 2, 3])


class PolicyTests(unittest.TestCase):
    def stream(self, n=6000, seed=3):
        rng = np.random.default_rng(seed)
        risk = rng.uniform(0.2, 0.6, n)
        return {"risk": risk, "y": (rng.random(n) < risk).astype(int), "amount": np.full(n, 300.0)}

    def test_learned_friction_moves_away_from_step_up_after_drift(self):
        s = self.stream()
        sim_at = drifting(SIM, 3000, 0.2)
        result, policy = run_thompson(s, sim_at, 0.1, 0, ThompsonFriction(SIM, 0.1, ACTIONS, gamma=0.995), 0.0)
        before = np.bincount(result["actions"][:3000], minlength=4)
        after = np.bincount(result["actions"][-1500:], minlength=4)
        self.assertGreater(before[1], before[2])  # step-up preferred while it works
        self.assertGreater(after[2] + after[3], after[1])  # hold/block once attackers pass step-up
        self.assertLess(policy.rates()["STEP_UP"]["catch"], 0.85)  # learned that step-up got worse

    def test_thompson_state_round_trips(self):
        policy = ThompsonFriction(SIM, 0.1, ACTIONS)
        policy.update(1, True, True, False)
        clone = ThompsonFriction.from_state(policy.state())
        mask = np.array([False, True, True, True])
        self.assertEqual(clone.choose(0.5, 300.0, mask), policy.choose(0.5, 300.0, mask))

    def test_linucb_learns_the_cheaper_arm(self):
        bandit = LinUCB(2, 4, alpha=0.1)
        x = context(0.5, 100.0)
        for _ in range(200):
            bandit.update(0, x, -1.0)
            bandit.update(1, x, -0.1)
        self.assertEqual(bandit.choose(x), 1)
        self.assertEqual(LinUCB.from_state(bandit.state()).choose(x), 1)

    def test_replay_only_reports_the_taken_action(self):
        s = self.stream(100)
        seen = []
        replay(s, lambda i: 1, lambda i: SIM, 0.1, 0, lambda i, a, outcome: seen.append(a))
        self.assertEqual(set(seen), {1})


class AttackerTests(unittest.TestCase):
    def test_mutations_stay_inside_the_threat_model(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            y = mutate(TX, rng, 0.5)
            self.assertGreaterEqual(y["TransactionAmt"], TX["TransactionAmt"] * 0.5 - 0.01)
            self.assertLessEqual(y["TransactionDT"] - TX["TransactionDT"], 23 * 3600)

    def test_more_queries_never_hurt_the_open_score_attacker(self):
        rows = [dict(TX, TransactionAmt=a) for a in (150.0, 220.0, 260.0)]
        rates = []
        for q in (2, 50):
            wins = sum(hill_climb(x, toy_risk, np.random.default_rng(0), q, 0.35)[2] for x in rows)
            rates.append(wins)
        self.assertLessEqual(rates[0], rates[1])
        self.assertTrue(random_search(TX, toy_risk, np.random.default_rng(0), 50, 0.35)[2])


if __name__ == "__main__":
    unittest.main()
