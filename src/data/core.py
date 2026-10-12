"""Row-local features shared by training, serving, counterfactuals and the attacker.

Numpy only, so the deployed API does not need pandas. Functions take any mapping of
column name -> array-like (a pandas DataFrame works too) and never look at other rows:
history and graph columns must already be present (see `src.data.features`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CATEGORICAL = ("ProductCD", "card4", "card6", "DeviceType", "P_emaildomain")
ENTITY_COLS = ("DeviceInfo", "P_emaildomain", "addr1")
GRAPH_FEATURES = tuple(f"{col}_{kind}" for col in ENTITY_COLS for kind in ("prior_cnt", "distinct_uids"))
HISTORY_FEATURES = ("prior_cnt", "prior_mean_amt", "card_device_prior")
ROW_FEATURES = ("log_amt", "amt_ratio", "hour", "dow", "since_last")
FEATURES = (
    "TransactionAmt", "dist1", *ROW_FEATURES, *HISTORY_FEATURES, *GRAPH_FEATURES, *CATEGORICAL,
)

# Plain-language names for explanations.
LABELS = {
    "TransactionAmt": "Transaction amount",
    "dist1": "Distance from billing address",
    "log_amt": "Transaction amount",
    "amt_ratio": "Amount vs. this card's usual spend",
    "hour": "Time of day",
    "dow": "Day of week",
    "since_last": "Time since this card's last payment",
    "prior_cnt": "Card history length",
    "prior_mean_amt": "Card's usual spend",
    "card_device_prior": "Card seen on this device before",
    "DeviceInfo_prior_cnt": "Device activity",
    "DeviceInfo_distinct_uids": "Cards sharing this device",
    "P_emaildomain_prior_cnt": "Email domain activity",
    "P_emaildomain_distinct_uids": "Cards sharing this email domain",
    "addr1_prior_cnt": "Billing region activity",
    "addr1_distinct_uids": "Cards in this billing region",
    "ProductCD": "Product type",
    "card4": "Card network",
    "card6": "Funding source (debit/credit)",
    "DeviceType": "Device type",
    "P_emaildomain": "Email domain",
}


def _num(cols: Mapping[str, Any], name: str) -> np.ndarray:
    return np.asarray(cols[name], dtype=float)


def row_features(cols: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Deterministic per-row features. `prev_dt` is the card's previous timestamp (NaN if none)."""
    dt = _num(cols, "TransactionDT")
    amt = _num(cols, "TransactionAmt")
    prev = _num(cols, "prev_dt")
    return {
        "hour": (dt // 3600) % 24,
        "dow": (dt // 86400) % 7,
        "log_amt": np.log1p(np.clip(amt, 0, None)),
        "amt_ratio": amt / (_num(cols, "prior_mean_amt") + 1e-6),
        "since_last": np.where(np.isnan(prev), -1.0, dt - prev),
    }


def encode(values: Any, levels: Sequence[str]) -> np.ndarray:
    """Map strings to their index in train-learned `levels`; unseen or missing -> -1."""
    index = {level: i for i, level in enumerate(levels)}
    out = np.empty(len(values), dtype=float)
    for i, value in enumerate(values):
        out[i] = -1.0 if value is None or value != value else index.get(str(value), -1)
    return out


def feature_matrix(
    cols: Mapping[str, Any],
    levels: Mapping[str, Sequence[str]],
    features: Sequence[str] = FEATURES,
) -> np.ndarray:
    """Build the model input matrix in a fixed column order."""
    derived = row_features(cols)
    columns = []
    for name in features:
        if name in derived:
            columns.append(derived[name])
        elif name in CATEGORICAL:
            columns.append(encode(list(cols[name]), levels[name]))
        else:
            columns.append(_num(cols, name))
    return np.column_stack(columns).astype(float)


def rows_to_columns(rows: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    """Turn a list of transaction dicts into a column mapping."""
    keys = rows[0].keys()
    return {key: [row.get(key) for row in rows] for key in keys}
