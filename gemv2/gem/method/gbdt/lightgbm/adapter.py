"""
LightGBM data adapter -- converts SplitView to lgb.Dataset.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...base.adapter import BaseAdapter
from ....core.data import ProcessedBundle, SplitView


class LightGBMAdapter(BaseAdapter):
    """Converts SplitView to lgb.Dataset."""

    def __init__(
        self,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        free_raw_data: bool = True,
    ):
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.free_raw_data = free_raw_data

    @staticmethod
    def _import_lgb() -> Any:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is required for LightGBMAdapter") from exc
        return lgb

    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Any:
        lgb = self._import_lgb()
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        feature_name = self.feature_name
        if feature_name == "auto" and view.feature_names:
            feature_name = view.feature_names

        return lgb.Dataset(
            data=view.X,
            label=y,
            weight=weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs,
        )

    def from_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        lgb = self._import_lgb()
        y_flat = y.ravel() if y.ndim > 1 else y
        feature_name = self.feature_name
        if feature_name == "auto" and feature_names:
            feature_name = feature_names

        return lgb.Dataset(
            data=X,
            label=y_flat,
            weight=weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs,
        )

    def to_train_val_test(
        self,
        views: ProcessedBundle,
        weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        weights = weights or {}
        dtrain = self.to_dataset(views.train, weight=weights.get("train"), **kwargs)
        dval = self.to_dataset(views.val, reference=dtrain, weight=weights.get("val"), **kwargs)
        dtest = self.to_dataset(views.test, reference=dtrain, weight=weights.get("test"), **kwargs)
        return dtrain, dval, dtest

    def create_datasets_dict(
        self,
        views: ProcessedBundle,
        weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        dtrain, dval, dtest = self.to_train_val_test(views, weights, **kwargs)
        return {"train": dtrain, "val": dval, "test": dtest}


__all__ = ["LightGBMAdapter"]
