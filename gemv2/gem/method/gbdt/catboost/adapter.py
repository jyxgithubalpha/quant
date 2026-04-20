"""
CatBoost data adapter -- converts SplitView to catboost.Pool.
"""

from typing import Any, List, Optional

import numpy as np

from ...base.adapter import BaseAdapter
from ....core.data import SplitView


class CatBoostAdapter(BaseAdapter):
    """Converts SplitView to catboost.Pool."""

    def __init__(self, cat_features: Optional[List[int]] = None):
        self.cat_features = cat_features or []

    @staticmethod
    def _import_catboost() -> Any:
        try:
            from catboost import Pool
        except ImportError as exc:
            raise ImportError("catboost is required for CatBoostAdapter") from exc
        return Pool

    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Any:
        Pool = self._import_catboost()
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        return Pool(
            data=view.X,
            label=y,
            weight=weight,
            feature_names=view.feature_names,
            cat_features=self.cat_features or None,
            **kwargs,
        )


__all__ = ["CatBoostAdapter"]
