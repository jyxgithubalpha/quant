"""
StatsCalculator -- compute descriptive statistics from train/val data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TransformStats:
    """Descriptive statistics computed from raw data before transforms."""

    X_mean: Optional[np.ndarray] = None
    X_std: Optional[np.ndarray] = None
    X_lower_quantile: Optional[np.ndarray] = None
    X_upper_quantile: Optional[np.ndarray] = None
    X_median: Optional[np.ndarray] = None
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    y_lower_quantile: Optional[np.ndarray] = None
    y_upper_quantile: Optional[np.ndarray] = None
    y_median: Optional[np.ndarray] = None
    custom: Dict[str, Any] = field(default_factory=dict)


class StatsCalculator:
    """Compute X and y statistics from train (optionally combined with val) data."""

    @staticmethod
    def compute(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        use_combined: bool = False,
    ) -> TransformStats:
        if use_combined and X_val is not None:
            X = np.vstack([X_train, X_val])
            y = np.concatenate([y_train.ravel(), y_val.ravel()])
        else:
            X = X_train
            y = y_train.ravel() if y_train.ndim > 1 else y_train

        return TransformStats(
            X_mean=np.nanmean(X, axis=0),
            X_std=np.nanstd(X, axis=0),
            X_lower_quantile=np.nanquantile(X, lower_quantile, axis=0),
            X_upper_quantile=np.nanquantile(X, upper_quantile, axis=0),
            X_median=np.nanmedian(X, axis=0),
            y_mean=np.array([np.nanmean(y)]),
            y_std=np.array([np.nanstd(y)]),
            y_lower_quantile=np.array([np.nanquantile(y, lower_quantile)]),
            y_upper_quantile=np.array([np.nanquantile(y, upper_quantile)]),
            y_median=np.array([np.nanmedian(y)]),
        )
