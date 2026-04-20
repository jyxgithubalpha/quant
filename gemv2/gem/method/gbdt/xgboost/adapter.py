"""
XGBoost data adapter -- converts SplitView to xgb.DMatrix.
"""

from typing import Any, Optional

import numpy as np

from ...base.adapter import BaseAdapter
from ....core.data import SplitView


class XGBoostAdapter(BaseAdapter):
    """Converts SplitView to xgb.DMatrix."""

    def __init__(self, enable_categorical: bool = False):
        self.enable_categorical = enable_categorical

    @staticmethod
    def _import_xgb() -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostAdapter") from exc
        return xgb

    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Any:
        xgb = self._import_xgb()
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        return xgb.DMatrix(
            data=view.X,
            label=y,
            weight=weight,
            feature_names=view.feature_names,
            enable_categorical=self.enable_categorical,
            **kwargs,
        )


__all__ = ["XGBoostAdapter"]
