"""Numpy-only scoring of an exported LightGBM model plus isotonic calibration.

Training uses LightGBM; the deployed API only needs numpy. `TreeEnsemble.from_lightgbm`
flattens the booster's trees into arrays, and `predict_raw` walks every tree for every row
at once. Parity with `Booster.predict` is checked in tests and at export time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_ZERO = 1e-35  # LightGBM's kZeroThreshold
_MISSING = {"None": 0, "Zero": 1, "NaN": 2}


class TreeEnsemble:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.feature = arrays["feature"].astype(np.int32)        # -1 for leaves
        self.threshold = arrays["threshold"].astype(np.float64)
        self.left = arrays["left"].astype(np.int32)
        self.right = arrays["right"].astype(np.int32)
        self.default_left = arrays["default_left"].astype(bool)
        self.missing = arrays["missing"].astype(np.int8)
        self.value = arrays["value"].astype(np.float64)          # leaf value or internal value
        self.roots = arrays["roots"].astype(np.int32)
        self.depth = int(arrays["depth"])

    # -- export ------------------------------------------------------------
    @classmethod
    def from_lightgbm(cls, booster: Any) -> "TreeEnsemble":
        dump = booster.dump_model()
        cols: dict[str, list] = {k: [] for k in ("feature", "threshold", "left", "right", "default_left", "missing", "value")}
        roots, max_depth = [], 0

        def add(node: dict, depth: int) -> int:
            nonlocal max_depth
            index = len(cols["feature"])
            for values in cols.values():
                values.append(None)
            if "leaf_value" in node:
                max_depth = max(max_depth, depth)
                cols["feature"][index], cols["threshold"][index] = -1, 0.0
                cols["left"][index] = cols["right"][index] = index
                cols["default_left"][index], cols["missing"][index] = False, 0
                cols["value"][index] = node["leaf_value"]
                return index
            if node.get("decision_type", "<=") != "<=":
                raise ValueError("Only numerical splits are supported.")
            cols["feature"][index] = node["split_feature"]
            cols["threshold"][index] = node["threshold"]
            cols["default_left"][index] = bool(node["default_left"])
            cols["missing"][index] = _MISSING[node["missing_type"]]
            cols["value"][index] = node["internal_value"]
            cols["left"][index] = add(node["left_child"], depth + 1)
            cols["right"][index] = add(node["right_child"], depth + 1)
            return index

        for tree in dump["tree_info"]:
            roots.append(add(tree["tree_structure"], 0))
        arrays = {k: np.array(v) for k, v in cols.items()}
        arrays["roots"] = np.array(roots)
        arrays["depth"] = np.array(max_depth)
        return cls(arrays)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "feature": self.feature, "threshold": self.threshold, "left": self.left, "right": self.right,
            "default_left": self.default_left, "missing": self.missing, "value": self.value,
            "roots": self.roots, "depth": np.array(self.depth),
        }

    # -- scoring -----------------------------------------------------------
    def _walk(self, X: np.ndarray, contrib: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        node = np.broadcast_to(self.roots, (n, len(self.roots))).copy()
        rows = np.broadcast_to(np.arange(n)[:, None], node.shape)
        for _ in range(self.depth):
            feat = self.feature[node]
            internal = feat >= 0
            if not internal.any():
                break
            x = X[rows, np.where(internal, feat, 0)]
            missing = self.missing[node]
            nan = np.isnan(x)
            x = np.where(nan & (missing != 2), 0.0, x)
            is_missing = ((missing == 1) & (np.abs(x) <= _ZERO)) | ((missing == 2) & nan)
            go_left = np.where(is_missing, self.default_left[node], x <= self.threshold[node])
            nxt = np.where(internal, np.where(go_left, self.left[node], self.right[node]), node)
            if contrib is not None:
                delta = np.where(internal, self.value[nxt] - self.value[node], 0.0)
                np.add.at(contrib, (rows[internal], feat[internal]), delta[internal])
            node = nxt
        return node

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.value[self._walk(X)].sum(axis=1)

    def contributions(self, X: np.ndarray, n_features: int) -> tuple[np.ndarray, float]:
        """Per-feature path contributions to the raw score (Saabas), plus the shared bias."""
        X = np.asarray(X, dtype=np.float64)
        contrib = np.zeros((X.shape[0], n_features))
        self._walk(X, contrib)
        return contrib, float(self.value[self.roots].sum())


class RiskModel:
    """Calibrated fraud probability: isotonic(sigmoid(trees(X)))."""

    def __init__(self, trees: TreeEnsemble, calib_x: np.ndarray, calib_y: np.ndarray):
        self.trees = trees
        self.calib_x = np.asarray(calib_x, dtype=float)
        self.calib_y = np.asarray(calib_y, dtype=float)

    def raw_probability(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.trees.predict_raw(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.interp(self.raw_probability(X), self.calib_x, self.calib_y)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.trees.arrays(), calib_x=self.calib_x, calib_y=self.calib_y)

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        return cls(TreeEnsemble(arrays), arrays["calib_x"], arrays["calib_y"])
