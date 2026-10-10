from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_history(
    frame: pd.DataFrame,
    *,
    fallback_amount: float,
) -> pd.DataFrame:
    """Add prior-only entity history to a time-sorted frame.

    Rows missing either card1 or addr1 are treated as distinct entities rather
    than being accidentally grouped together as a shared "nan_nan" card.
    """
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"History features require columns: {', '.join(sorted(missing))}")
    if not frame["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted by TransactionDT before adding history.")
    if not np.isfinite(fallback_amount) or fallback_amount <= 0:
        raise ValueError("fallback_amount must be a finite positive training-set median.")

    result = frame.copy()
    has_entity = result["card1"].notna() & result["addr1"].notna()
    card = result["card1"].astype("string")
    address = result["addr1"].astype("string")
    entity = card + "_" + address
    entity.loc[~has_entity] = "missing_" + result.loc[
        ~has_entity, "TransactionID"
    ].astype("string")
    result["_entity_key"] = entity

    grouped = result.groupby("_entity_key", sort=False)
    prior_count = grouped.cumcount()
    prior_amount_sum = grouped["TransactionAmt"].cumsum() - result["TransactionAmt"]
    prior_mean = prior_amount_sum.div(prior_count.replace(0, np.nan))
    result["prior_cnt"] = prior_count.astype("int64")
    result["prior_mean_amt"] = prior_mean.fillna(fallback_amount)
    result["since_last"] = grouped["TransactionDT"].diff().fillna(-1)
    return result.drop(columns="_entity_key")


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic row-local features using already-computed history."""
    required = {"TransactionDT", "TransactionAmt", "prior_mean_amt"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"make_features requires columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["hour"] = (result["TransactionDT"] // 3600) % 24
    result["dow"] = (result["TransactionDT"] // 86400) % 7
    result["log_amt"] = np.log1p(result["TransactionAmt"].clip(lower=0))
    result["amt_ratio"] = result["TransactionAmt"] / (
        result["prior_mean_amt"] + 1e-6
    )
    return result


def learn_category_levels(
    train: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, list[str]]:
    """Learn categorical vocabularies on train only for use on later splits."""
    levels: dict[str, list[str]] = {}
    for column in columns:
        if column not in train:
            raise ValueError(f"Categorical column {column!r} is not in the train split.")
        levels[column] = sorted(train[column].dropna().astype(str).unique().tolist())
    return levels


def apply_category_levels(
    frame: pd.DataFrame,
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply train-derived categorical levels, mapping unseen values to missing."""
    result = frame.copy()
    for column, values in levels.items():
        if column not in result:
            raise ValueError(f"Categorical column {column!r} is absent from this split.")
        observed = result[column].astype("string")
        observed = observed.where(observed.isin(values), pd.NA)
        result[column] = pd.Categorical(
            observed,
            categories=values,
        )
    return result
