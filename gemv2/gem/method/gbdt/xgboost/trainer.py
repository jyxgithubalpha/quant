"""
XGBoost trainer.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...base.trainer import BaseTrainer
from ....core.data import ProcessedBundle
from ....core.training import FitResult, TrainConfig
from .adapter import XGBoostAdapter


class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer."""

    def __init__(
        self,
        adapter: Optional[XGBoostAdapter] = None,
        use_gpu: Optional[bool] = None,
    ):
        self.adapter = adapter or XGBoostAdapter()
        self.use_gpu = use_gpu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        views: ProcessedBundle,
        config: TrainConfig,
        phase: str = "full",
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> FitResult:
        return self._fit_local(views, config, phase, sample_weights)

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        best = getattr(model, "best_iteration", None)
        if best is not None:
            best = int(best)
            return model.inplace_predict(X, iteration_range=(0, best + 1))
        return model.inplace_predict(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_xgboost() -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostTrainer") from exc
        return xgb

    def _apply_gpu_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = bool(params.get("use_gpu", False))
        params.pop("use_gpu", None)
        if use_gpu:
            params.setdefault("tree_method", "gpu_hist")
            params.setdefault("predictor", "gpu_predictor")
            params.setdefault("device", "cuda")
        return params

    def _build_datasets(
        self,
        views: ProcessedBundle,
        sample_weights: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[Any, Any]:
        weights = sample_weights or {}
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, weight=weights.get("val"))
        return dtrain, dval

    @staticmethod
    def _resolve_best_iteration(
        model: Any,
        evals_result: Dict[str, Dict[str, list]],
        fallback: int,
    ) -> int:
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            best_iteration = int(best_iteration)
            if best_iteration >= 0:
                return min(best_iteration + 1, max(1, fallback))

        for split_metrics in evals_result.values():
            for series in split_metrics.values():
                if isinstance(series, list) and series:
                    return min(len(series), max(1, fallback))
        return max(1, fallback)

    def _fit_local(
        self,
        views: ProcessedBundle,
        config: TrainConfig,
        phase: str,
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> FitResult:
        xgb = self._import_xgboost()

        dtrain, dval = self._build_datasets(views, sample_weights)
        params = dict(config.params)
        params["seed"] = config.seed
        if "eval_metric" not in params and config.monitor_metrics:
            params["eval_metric"] = config.monitor_metrics[0]
        params = self._apply_gpu_params(params)

        evals_result: Dict[str, Dict[str, list]] = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=config.max_iterations,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=config.early_stopping_patience,
            verbose_eval=config.log_interval if phase == "full" else False,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.max_iterations)
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
        )


__all__ = ["XGBoostTrainer"]
