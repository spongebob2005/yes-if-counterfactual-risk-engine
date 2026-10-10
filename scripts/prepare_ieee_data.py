import json
from pathlib import Path

import pandas as pd

from src.data.features import (
    add_history,
    apply_category_levels,
    learn_category_levels,
    make_features,
)
from src.data.load import (
    SPLIT_NAMES,
    assert_time_ordered_splits,
    load_ieee,
    print_fraud_rates,
    save_splits,
    time_split,
)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CATEGORICAL_COLUMNS = ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain")


def main() -> None:
    frame = load_ieee(RAW_DIR)
    if "isFraud" not in frame:
        raise ValueError("The IEEE-CIS training transactions must contain isFraud.")
    raw_splits = time_split(frame)
    assert_time_ordered_splits(raw_splits)

    fallback_amount = float(raw_splits["train"]["TransactionAmt"].median())
    with_history = add_history(frame, fallback_amount=fallback_amount)
    featured = make_features(with_history)
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
