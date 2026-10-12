import unittest

import numpy as np

from src.models.metrics import ece, recall_at_fpr
from src.models.serving import RiskModel, TreeEnsemble

try:
    import lightgbm  # noqa: F401
    from sklearn.isotonic import IsotonicRegression

    from src.models.detector import check_parity, choose_thresholds, export, fit_detector
    HAVE_TRAINING = True
except ImportError:  # the deployed API does not install training dependencies
    HAVE_TRAINING = False

PARAMS = {"learning_rate": 0.1, "num_leaves": 15, "n_estimators": 60, "early_stopping": 20,
          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20}
SIM = {
    "ALLOW": {"fric": 0.0, "catch": 0.0, "abandon": 0.0},
    "STEP_UP": {"fric": 2.0, "catch": 0.85, "abandon": 0.03},
    "HOLD": {"fric": 6.0, "catch": 0.9, "abandon": 0.15},
    "BLOCK": {"fric": 10.0, "catch": 1.0, "abandon": 1.0},
}


def toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[rng.random(n) < 0.05, 2] = np.nan  # missing values take LightGBM's default branch
    logit = 2 * X[:, 0] - X[:, 1] + np.nan_to_num(X[:, 2]) - 3
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@unittest.skipUnless(HAVE_TRAINING, "training dependencies not installed")
class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = toy()
        cls.X, cls.y = X, y
        cls.model, cls.iso = fit_detector(X[:1800], y[:1800], X[1800:2400], y[1800:2400], X[2400:], y[2400:], PARAMS)
        cls.risk = export(cls.model, cls.iso)

    def test_numpy_predictor_matches_lightgbm_exactly(self):
        self.assertLessEqual(check_parity(self.model, self.iso, self.risk, self.X), 1e-9)

    def test_contributions_add_up_to_the_raw_score(self):
        contrib, bias = self.risk.trees.contributions(self.X[:50], 4)
        np.testing.assert_allclose(contrib.sum(axis=1) + bias, self.risk.trees.predict_raw(self.X[:50]), atol=1e-9)
        self.assertGreater(np.abs(contrib[:, 0]).mean(), np.abs(contrib[:, 3]).mean())  # X0 matters, X3 is noise

    def test_save_and_load_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npz"
            self.risk.save(path)
            np.testing.assert_array_equal(RiskModel.load(path).predict(self.X), self.risk.predict(self.X))

    def test_thresholds_form_an_ordered_ladder(self):
        p = self.risk.predict(self.X[2400:])
        t = choose_thresholds(p, self.y[2400:], np.full(len(p), 100.0), SIM, 0.1)
        self.assertLessEqual(t["tau_allow"], t["tau_hold"])
        self.assertLessEqual(t["tau_hold"], t["tau_block"])


class MetricTests(unittest.TestCase):
    def test_perfectly_calibrated_scores_have_low_ece(self):
        rng = np.random.default_rng(1)
        p = rng.random(20000)
        y = (rng.random(20000) < p).astype(int)
        self.assertLess(ece(y, p), 0.02)
        self.assertGreater(ece(y, np.clip(p + 0.3, 0, 1)), 0.2)

    def test_recall_at_fpr_of_a_perfect_ranker_is_one(self):
        y = np.array([0, 0, 0, 1, 1])
        self.assertEqual(recall_at_fpr(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])), 1.0)

    def test_single_leaf_trees_are_supported(self):
        arrays = {"feature": np.array([-1]), "threshold": np.array([0.0]), "left": np.array([0]),
                  "right": np.array([0]), "default_left": np.array([False]), "missing": np.array([0]),
                  "value": np.array([0.5]), "roots": np.array([0]), "depth": np.array(0)}
        self.assertEqual(TreeEnsemble(arrays).predict_raw(np.zeros((3, 2))).tolist(), [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
