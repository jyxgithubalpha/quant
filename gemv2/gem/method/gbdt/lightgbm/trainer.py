"""
LightGBM trainer.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...base.trainer import BaseTrainer
from ....core.data import ProcessedBundle
from ....core.training import FitResult, TrainConfig
from .adapter import LightGBMAdapter


class LightGBMTrainer(BaseTrainer):
    """LightGBM trainer."""

    def __init__(self, adapter: Optional[LightGBMAdapter] = None):
        self.adapter = adapter or LightGBMAdapter()

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
        best = int(getattr(model, "best_iteration", 0) or 0)
        num_iter = best if best > 0 else None
        return model.predict(X, num_iteration=num_iter)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_lightgbm() -> Any:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is required for LightGBMTrainer") from exc
        return lgb

    @staticmethod
    def _resolve_best_iteration(
        model: Any,
        evals_result: Dict[str, Dict[str, list]],
        fallback: int,
    ) -> int:
        best_iteration = int(getattr(model, "best_iteration", 0) or 0)
        if best_iteration > 0:
            return min(best_iteration, max(1, fallback))

        for split_metrics in evals_result.values():
            for series in split_metrics.values():
                if isinstance(series, list) and series:
                    return min(len(series), max(1, fallback))

        return max(1, fallback)

    def _build_datasets(
        self,
        views: ProcessedBundle,
        sample_weights: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        weights = sample_weights or {}
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, reference=dtrain, weight=weights.get("val"))
        return dtrain, dval, {"train": dtrain, "val": dval}

    def _build_callbacks(
        self,
        lgb: Any,
        config: TrainConfig,
        phase: str,
        evals_result: Dict[str, Dict[str, list]],
    ) -> list:
        verbose = phase == "full"
        return [
            lgb.early_stopping(
                stopping_rounds=config.early_stopping_patience,
                first_metric_only=True,
                verbose=verbose,
            ),
            lgb.log_evaluation(period=config.log_interval if verbose else 0),
            lgb.record_evaluation(evals_result),
        ]

    def _train_with_lgb(
        self,
        lgb: Any,
        params: Dict[str, Any],
        dtrain: Any,
        dval: Any,
        config: TrainConfig,
        phase: str,
        feval_list: Optional[list] = None,
    ) -> Tuple[Any, Dict[str, Dict[str, list]], int]:
        evals_result: Dict[str, Dict[str, list]] = {}
        callbacks = self._build_callbacks(lgb, config, phase, evals_result)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=config.max_iterations,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            feval=feval_list if feval_list else None,
            callbacks=callbacks,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.max_iterations)
        return model, evals_result, best_iteration

    def _fit_local(
        self,
        views: ProcessedBundle,
        config: TrainConfig,
        phase: str = "full",
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> FitResult:
        lgb = self._import_lightgbm()
        from .feval import FevalAdapterFactory, ObjectiveFactory

        dtrain, dval, datasets = self._build_datasets(views, sample_weights)

        params = dict(config.params)
        params["seed"] = config.seed

        objective = ObjectiveFactory.get(
            config.monitor_metrics[0] if config.monitor_metrics else "regression",
            views=views,
            datasets=datasets,
        )
        params["objective"] = objective

        split_views = {"train": views.train, "val": views.val, "test": views.test}
        feval_list = FevalAdapterFactory.create(config.monitor_metrics, split_views, datasets)

        model, evals_result, best_iteration = self._train_with_lgb(
            lgb=lgb,
            params=params,
            dtrain=dtrain,
            dval=dval,
            config=config,
            phase=phase,
            feval_list=feval_list,
        )

        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
        )


__all__ = ["LightGBMTrainer"]
