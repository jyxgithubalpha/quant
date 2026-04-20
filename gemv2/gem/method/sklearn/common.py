"""
Generic sklearn trainer and importance extractor for tabular regression models.
"""

import inspect
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import polars as pl

from ...core.data import ProcessedBundle
from ...core.training import FitResult, TrainConfig
from ..base.trainer import BaseTrainer
from ..base.importance import BaseImportanceExtractor


class SklearnTrainer(BaseTrainer):
    """
    Generic sklearn trainer.

    Can be used with any sklearn-compatible estimator by passing ``model_cls``
    and ``default_params`` directly, or by subclassing and overriding them.

    Alternatively, pass ``model_config`` (a dict with ``_target_`` and optional
    hyperparameters) to derive ``model_cls`` and ``default_params`` automatically.
    This is the path taken when the method config has a ``model:`` sub-key.
    """

    def __init__(
        self,
        model_cls: Optional[type] = None,
        default_params: Optional[Dict[str, Any]] = None,
        model_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if model_config is not None and model_cls is None:
            model_cls, default_params = self._parse_model_config(model_config, default_params)
        self.model_cls = model_cls
        self.default_params = default_params or {}

    @staticmethod
    def _parse_model_config(
        model_config: Mapping[str, Any],
        default_params: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[type], Dict[str, Any]]:
        """Extract model class and default params from a Hydra-style model config dict."""
        import importlib

        target = model_config.get("_target_")
        if not target:
            return None, dict(default_params or {})

        module_path, cls_name = str(target).rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)

        params = {k: v for k, v in model_config.items() if not k.startswith("_")}
        merged = {**params, **(default_params or {})}
        return cls, merged

    def fit(
        self,
        views: ProcessedBundle,
        config: TrainConfig,
        phase: str = "full",
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> FitResult:
        model = self._build_model(config)

        X_train = views.train.X
        y_train = views.train.y.ravel() if views.train.y.ndim > 1 else views.train.y
        weights = (sample_weights or {}).get("train")

        fit_kwargs: Dict[str, Any] = {}
        if weights is not None and self._supports_arg(model.fit, "sample_weight"):
            fit_kwargs["sample_weight"] = weights

        model.fit(X_train, y_train, **fit_kwargs)

        return FitResult(
            model=model,
            evals_result={"train": {}, "val": {}},
            best_iteration=1,
            params=dict(config.params),
            seed=config.seed,
        )

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict(X).ravel()

    def _build_model(self, config: TrainConfig) -> Any:
        if self.model_cls is None:
            raise ValueError(
                "model_cls must be set before calling fit(). "
                "Pass it to __init__ or subclass SklearnTrainer."
            )
        params = {**self.default_params, **dict(config.params or {})}
        if self._cls_supports_random_state(self.model_cls) and "random_state" not in params:
            params["random_state"] = config.seed
        return self.model_cls(**params)

    @staticmethod
    def _cls_supports_random_state(cls: type) -> bool:
        try:
            return "random_state" in inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _supports_arg(func: Any, arg_name: str) -> bool:
        try:
            return arg_name in inspect.signature(func).parameters
        except (TypeError, ValueError):
            return False


class SklearnImportanceExtractor(BaseImportanceExtractor):
    """
    Feature importance extractor for sklearn linear and tree-based models.

    Uses ``coef_`` for linear models and ``feature_importances_`` for tree models.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def extract(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, Optional[pl.DataFrame]]:
        importance = self._extract_raw_importance(model, len(feature_names))
        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)
        df = pl.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort("importance", descending=True)
        return importance, df

    @staticmethod
    def _extract_raw_importance(model: Any, n_features: int) -> np.ndarray:
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importance = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
            if importance.shape[0] != n_features:
                return np.zeros(n_features, dtype=np.float32)
            return importance.astype(np.float32)

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=np.float32)
            if importance.shape[0] != n_features:
                return np.zeros(n_features, dtype=np.float32)
            return importance

        return np.zeros(n_features, dtype=np.float32)
