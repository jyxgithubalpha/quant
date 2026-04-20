"""
MinSampleFilter -- set features/labels to NaN for date groups with too few samples.
"""

from typing import Optional, Tuple

import numpy as np

from .base import BaseTransform


class MinSampleFilter(BaseTransform):
    """
    Filter out date groups with sample count below *min_samples*.

    Sets X and y to NaN for those groups; downstream transforms or
    evaluation logic should handle NaN values accordingly.
    """

    def __init__(self, min_samples: int = 5) -> None:
        super().__init__()
        self.min_samples = min_samples
        self._filtered_dates: list = []

    def fit(self, X, y, keys=None):
        self._filtered_dates = []
        if keys is not None:
            unique_keys, counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique_keys, counts):
                if count < self.min_samples:
                    self._filtered_dates.append(key)
        return self

    def transform(self, X, y, keys=None):
        X_out = X.copy()
        y_out = y.copy()
        if keys is not None:
            unique_keys, counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique_keys, counts):
                if count < self.min_samples:
                    mask = keys == key
                    X_out[mask] = np.nan
                    y_out[mask] = np.nan
        return X_out, y_out
