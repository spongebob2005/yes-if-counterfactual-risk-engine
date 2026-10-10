from collections.abc import Sequence
from pathlib import Path

import pandas as pd


SPLIT_NAMES = ("train", "val", "calib", "test")
SPLIT_FRACTIONS = (0.60, 0.15, 0.10, 0.15)


def load_ieee(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Read and time-sort the IEEE-CIS training transaction and identity files."""
    root = Path(data_dir)
    transaction_path = root / "train_transaction.csv"
    identity_path = root / "train_identity.csv"
    missing = [path.name for path in (transaction_path, identity_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing IEEE-CIS file(s) in {root}: {', '.join(missing)}. "
            "Download the competition data and extract both CSV files first."
        )

    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    if "TransactionID" not in transactions or "TransactionID" not in identity:
        raise ValueError("Both IEEE-CIS CSV files must contain TransactionID.")
    if "TransactionDT" not in transactions:
        raise ValueError("The transaction CSV must contain TransactionDT.")

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(
        ["TransactionDT", "TransactionID"], kind="mergesort"
    ).reset_index(drop=True)


def time_split(
    frame: pd.DataFrame,
    fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> dict[str, pd.DataFrame]:
    """Split an already time-sorted frame without dropping remainder rows."""
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split fractions.")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("Every split fraction must be positive.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")
    if "TransactionDT" not in frame:
        raise ValueError("Frame must contain TransactionDT.")
    if not frame["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted by TransactionDT before splitting.")

    row_count = len(frame)
    timestamps = frame["TransactionDT"].reset_index(drop=True)
    boundaries = []
    for index in range(1, len(fractions)):
        boundary = int(row_count * sum(fractions[:index]))
        if boundaries and boundary <= boundaries[-1]:
            boundary = boundaries[-1] + 1
        if boundary <= 0 or boundary >= row_count:
            raise ValueError("Not enough rows to create four non-empty time splits.")
        if timestamps.iloc[boundary - 1] == timestamps.iloc[boundary]:
            boundary_time = timestamps.iloc[boundary]
            boundary = int(timestamps.searchsorted(boundary_time, side="right"))
        if boundaries and boundary <= boundaries[-1]:
            raise ValueError(
                "Repeated transaction times prevent four non-empty chronological splits."
            )
        boundaries.append(boundary)
    edges = [0, *boundaries, row_count]
    splits = {
        name: frame.iloc[edges[index] : edges[index + 1]].copy()
        for index, name in enumerate(SPLIT_NAMES)
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("Not enough distinct transaction times for four non-empty splits.")
    return splits


def assert_time_ordered_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Assert that all non-empty sequential splits move forward in time."""
    previous_max: float | int | None = None
    for name in SPLIT_NAMES:
        part = splits[name]
        if part.empty:
            raise AssertionError(f"The {name} split is empty.")
        current_min = part["TransactionDT"].min()
        current_max = part["TransactionDT"].max()
        if previous_max is not None and current_min <= previous_max:
            raise AssertionError(f"The {name} split is not strictly later than prior splits.")
        previous_max = current_max


def print_fraud_rates(splits: dict[str, pd.DataFrame]) -> None:
    """Print row counts and positive-label rates for each prepared split."""
    if any("isFraud" not in part for part in splits.values()):
        raise ValueError("Every split must contain the isFraud target column.")
    for name in SPLIT_NAMES:
        part = splits[name]
        rate = part["isFraud"].mean() * 100
        print(f"{name:>5}: {len(part):,} rows, fraud rate {rate:.2f}%")


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path = "data/processed",
) -> None:
    """Write the four prepared splits as Parquet files."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        splits[name].to_parquet(root / f"{name}.parquet", index=False)
