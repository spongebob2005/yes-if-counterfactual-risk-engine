import json
from pathlib import Path

import pandas as pd

from src.data.core import CATEGORICAL, IEEE_CATEGORICAL
from src.data.features import (
    add_graph_feats,
    add_history,
    apply_category_levels,
    learn_category_levels,
    make_features,
)
from src.data.load import (
    IEEE_COLUMNS,
    SPLIT_NAMES,
    assert_time_ordered_splits,
    load_ieee,
    print_fraud_rates,
    save_splits,
    time_split,
)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
# The same columns and categoricals `python -m src.experiments.run_all --source ieee` trains on.
CATEGORICAL_COLUMNS = (*CATEGORICAL, *IEEE_CATEGORICAL)


def main() -> None:
    frame = load_ieee(RAW_DIR, IEEE_COLUMNS)
    if "isFraud" not in frame:
        raise ValueError("The IEEE-CIS training transactions must contain isFraud.")
    raw_splits = time_split(frame)
    assert_time_ordered_splits(raw_splits)

    fallback_amount = float(raw_splits["train"]["TransactionAmt"].median())
    with_history = add_history(frame, fallback_amount=fallback_amount)
    featured = make_features(add_graph_feats(with_history))
    featured_splits = time_split(featured)

    category_levels = learn_category_levels(
        featured_splits["train"], CATEGORICAL_COLUMNS
    )
    encoded_splits = {
        name: apply_category_levels(featured_splits[name], category_levels)
        for name in SPLIT_NAMES
    }
    assert_time_ordered_splits(encoded_splits)
    print_fraud_rates(encoded_splits)
    save_splits(encoded_splits, PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "category_levels.json").write_text(
        json.dumps(category_levels, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved train, val, calib, and test Parquet files to {PROCESSED_DIR}.")


if __name__ == "__main__":
    main()
