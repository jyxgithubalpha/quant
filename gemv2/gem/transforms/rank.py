"""
RankNormalize -- rank transformation (optionally normalized to [0, 1]).
"""

from typing import Literal, Optional, Tuple

import numpy as np

from .base import BaseTransform


class RankNormalize(BaseTransform):
    """Convert values to ranks, optionally normalized to [0, 1]."""

    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        per_date: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.target = target
        self.per_date = per_date
        self.normalize = normalize

    def fit(self, X, y, keys=None):
        return self

    @staticmethod
    def _rank_1d(x: np.ndarray, normalize: bool) -> np.ndarray:
        ranks = np.argsort(np.argsort(x)).astype(np.float32)
        if normalize and len(x) > 1:
            ranks = ranks / (len(x) - 1)
        return ranks

    def _rank_array(self, arr, keys):
        out = np.zeros_like(arr, dtype=np.float32)
        if self.per_date and keys is not None:
            for key in np.unique(keys):
                mask = keys == key
                sub = arr[mask]
                if sub.ndim > 1:
                    for i in range(sub.shape[1]):
                        out[mask, i] = self._rank_1d(sub[:, i], self.normalize)
                else:
                    out[mask] = self._rank_1d(sub, self.normalize)
        else:
            if arr.ndim > 1:
                for i in range(arr.shape[1]):
                    out[:, i] = self._rank_1d(arr[:, i], self.normalize)
            else:
                out = self._rank_1d(arr, self.normalize)
        return out

    def transform(self, X, y, keys=None):
        X_out = X.copy()
        y_out = y.copy()
        if self.target in ("X", "both"):
            X_out = self._rank_array(X_out, keys)
        if self.target in ("y", "both"):
            y_out = self._rank_array(y_out, keys)
        return X_out, y_out
