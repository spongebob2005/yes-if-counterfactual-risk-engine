from collections.abc import Iterable

import numpy as np
import pandas as pd

from .core import ENTITY_COLS, row_features


def add_history(
    frame: pd.DataFrame,
    *,
    fallback_amount: float,
) -> pd.DataFrame:
    """Add prior-only entity history to a time-sorted frame.

    Rows missing either card1 or addr1 are treated as distinct entities rather
    than being accidentally grouped together as a shared "nan_nan" card.
    `prev_dt` is kept raw so `since_last` can be recomputed when a counterfactual
    moves the payment later in time.
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
    result["prev_dt"] = grouped["TransactionDT"].shift(1).astype(float)
    if "DeviceInfo" in result:
        has_device = result["DeviceInfo"].notna()
        device = result["DeviceInfo"].astype("string").fillna("")
        pair_count = result.groupby([entity, device], sort=False).cumcount()
        result["card_device_prior"] = pair_count.where(has_device, 0).astype("int64")
    else:
        result["card_device_prior"] = 0
    return result.drop(columns="_entity_key")


def add_graph_feats(frame: pd.DataFrame, entity_cols: Iterable[str] = ENTITY_COLS) -> pd.DataFrame:
    """Shared-entity features (M3 level 1): for each device, email domain and billing
    region, how many earlier payments and how many distinct cards have touched it so far.

    Causal: computed on the time-sorted frame, each row counts itself and the past only.
    """
    if not frame["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted by TransactionDT before adding graph features.")
    result = frame.copy()
    uid = result["card1"].astype("string") + "_" + result["addr1"].astype("string")
    for col in entity_cols:
        present = result[col].notna()
        key = result[col].astype("string")
        prior = result.groupby(key, sort=False).cumcount()
        pair_new = (~pd.DataFrame({"k": key, "u": uid}).duplicated()).astype("int64")
        distinct = pair_new.groupby(key, sort=False).cumsum()
        result[f"{col}_prior_cnt"] = prior.where(present, 0).astype(float)
        result[f"{col}_distinct_uids"] = distinct.where(present, 0).astype(float)
    return result


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic row-local features using already-computed history.

    A thin pandas wrapper over `core.row_features`, which the API calls directly.
    """
    required = {"TransactionDT", "TransactionAmt", "prior_mean_amt"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"make_features requires columns: {', '.join(sorted(missing))}")
    columns = frame if "prev_dt" in frame else frame.assign(prev_dt=np.nan)
    return frame.assign(**row_features(columns))


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
