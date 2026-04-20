"""
ZScoreStandardize -- classical mean/std standardization.
"""

from typing import Literal, Optional, Tuple

import numpy as np

from .base import BaseTransform


class ZScoreStandardize(BaseTransform):
    """Z-score standardization: (x - mean) / std."""

    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        eps: float = 1e-8,
        per_date: bool = True,
    ) -> None:
        super().__init__()
        self.target = target
        self.eps = eps
        self.per_date = per_date
        self._mean_X: Optional[np.ndarray] = None
        self._std_X: Optional[np.ndarray] = None
        self._mean_y: Optional[np.ndarray] = None
        self._std_y: Optional[np.ndarray] = None

    def fit(self, X, y, keys=None):
        if not self.per_date or keys is None:
            self._mean_X = np.nanmean(X, axis=0)
            self._std_X = np.nanstd(X, axis=0)
            if y.ndim > 1:
                self._mean_y = np.nanmean(y, axis=0)
                self._std_y = np.nanstd(y, axis=0)
            else:
                self._mean_y = np.array([np.nanmean(y)])
                self._std_y = np.array([np.nanstd(y)])
        return self

    def _standardize_array(self, arr, mean, std, keys):
        out = arr.copy()
        if self.per_date and keys is not None:
            for key in np.unique(keys):
                mask = keys == key
                sub = out[mask]
                m = np.nanmean(sub, axis=0)
                s = np.nanstd(sub, axis=0) + self.eps
                out[mask] = (sub - m) / s
        else:
            out = (out - mean) / (std + self.eps)
        return out

    def transform(self, X, y, keys=None):
        X_out = X.copy()
        y_out = y.copy()
        if self.target in ("X", "both"):
            X_out = self._standardize_array(X_out, self._mean_X, self._std_X, keys)
        if self.target in ("y", "both"):
            y_out = self._standardize_array(y_out, self._mean_y, self._std_y, keys)
        return X_out, y_out

    def inverse_transform(self, X, y, keys=None):
        X_out = X.copy()
        y_out = y.copy()
        if not self.per_date or keys is None:
            if self.target in ("X", "both"):
                X_out = X_out * (self._std_X + self.eps) + self._mean_X
            if self.target in ("y", "both"):
                y_out = y_out * (self._std_y + self.eps) + self._mean_y
        return X_out, y_out
