import unittest

import pandas as pd

from src.data.features import (
    add_history,
    apply_category_levels,
    learn_category_levels,
    make_features,
)
from src.data.load import assert_time_ordered_splits, time_split


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "TransactionDT": list(range(0, 100, 10)),
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "card1": [1, 1, 2, 1, None, None, 1, 2, 1, 1],
            "addr1": [5, 5, 5, 5, None, None, 5, 5, 5, 5],
            "ProductCD": ["W", "W", "C", "W", "R", "R", "W", "C", "W", "W"],
            "isFraud": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = sample_frame()

    def test_time_split_is_chronological_and_preserves_all_rows(self):
        splits = time_split(self.frame)

        assert_time_ordered_splits(splits)
        self.assertEqual(sum(len(part) for part in splits.values()), len(self.frame))
        self.assertEqual(
            [len(splits[name]) for name in ("train", "val", "calib", "test")],
            [6, 1, 1, 2],
        )
        self.assertLess(
            splits["train"]["TransactionDT"].max(),
            splits["val"]["TransactionDT"].min(),
        )

    def test_split_boundary_does_not_separate_rows_with_the_same_timestamp(self):
        frame = self.frame.copy()
        frame.loc[6, "TransactionDT"] = frame.loc[5, "TransactionDT"]
        frame = frame.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)

        splits = time_split(frame)

        assert_time_ordered_splits(splits)
        self.assertEqual(sum(len(part) for part in splits.values()), len(frame))
        self.assertEqual(splits["train"]["TransactionDT"].max(), 50)
        self.assertGreater(splits["val"]["TransactionDT"].min(), 50)

    def test_history_uses_only_prior_entity_transactions(self):
        history = add_history(self.frame, fallback_amount=55.0)

        self.assertEqual(history.loc[0, "prior_cnt"], 0)
        self.assertEqual(history.loc[0, "prior_mean_amt"], 55.0)
        self.assertEqual(history.loc[1, "prior_cnt"], 1)
        self.assertEqual(history.loc[1, "prior_mean_amt"], 10.0)
        self.assertEqual(history.loc[3, "prior_mean_amt"], 15.0)
        self.assertEqual(history.loc[4, "prior_cnt"], 0)
        self.assertEqual(history.loc[5, "prior_cnt"], 0)

    def test_row_local_features_match_single_row_and_full_frame(self):
        history = add_history(self.frame, fallback_amount=55.0)
        full_features = make_features(history)
        single_features = make_features(history.iloc[[3]])

        pd.testing.assert_series_equal(
            full_features.iloc[3][["hour", "dow", "log_amt", "amt_ratio"]],
            single_features.iloc[0][["hour", "dow", "log_amt", "amt_ratio"]],
        )

    def test_categories_are_learned_on_train_and_unknowns_become_missing(self):
        levels = learn_category_levels(self.frame.iloc[:4], ["ProductCD"])
        transformed = apply_category_levels(self.frame.iloc[4:], levels)

        self.assertEqual(levels["ProductCD"], ["C", "W"])
        self.assertTrue(pd.isna(transformed.iloc[0]["ProductCD"]))


if __name__ == "__main__":
    unittest.main()
