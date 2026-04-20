"""
LightGBM feval (custom evaluation function) adapters and objective factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import numpy as np

from ....core.data import SplitView
from ...evaluators.metrics import Metric, MetricRegistry

if TYPE_CHECKING:
    import lightgbm as lgb


# =============================================================================
# Feval adapter
# =============================================================================


class FevalAdapter:
    """Wraps a Metric into the LightGBM feval callable interface."""

    def __init__(
        self,
        metric: Metric,
        split_data: Dict[str, SplitView],
        dataset_to_bundle: Dict[int, str],
    ):
        self.metric = metric
        self.split_data = split_data
        self.dataset_to_bundle = dataset_to_bundle

    def __call__(self, y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
        bundle_name = self.dataset_to_bundle.get(id(dataset))
        if bundle_name is None:
            raise ValueError("Unknown dataset passed to FevalAdapter.")

        view = self.split_data.get(bundle_name)
        score = self.metric.compute(y_pred, view)
        return self.metric.name, score, self.metric.higher_is_better


class FevalAdapterFactory:
    """Factory that builds a list of LightGBM feval callables from metric names."""

    @staticmethod
    def create(
        metric_names: List[str],
        split_data: Dict[str, SplitView],
        datasets: Dict[str, lgb.Dataset],
    ) -> List[Callable]:
        dataset_to_bundle = {id(ds): name for name, ds in datasets.items()}
        adapters = []
        for name in metric_names:
            metric = MetricRegistry.get(name)
            adapters.append(FevalAdapter(metric, split_data, dataset_to_bundle))
        return adapters


# =============================================================================
# Objective factory
# =============================================================================


class ObjectiveFactory:
    """Registry of custom LightGBM objective functions."""

    _registry: Dict[str, Callable] = {}
    _builtin: List[str] = [
        "regression", "regression_l2", "l2", "mse", "rmse",
        "regression_l1", "l1", "mae", "huber", "fair",
        "poisson", "binary", "multiclass", "cross_entropy", "lambdarank",
    ]

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str, **context) -> Union[str, Callable]:
        if name in cls._builtin:
            return name
        if name in cls._registry:
            return cls._registry[name](**context)
        import logging
        logging.getLogger(__name__).warning(
            "Objective '%s' not found, falling back to 'regression'.", name,
        )
        return "regression"


@ObjectiveFactory.register("pearsonr_ic_loss")
def _pearsonr_ic_loss_factory(**context) -> Callable:
    views = context["views"]
    datasets = context["datasets"]

    dataset_to_view = {
        id(datasets["train"]): views.train,
        id(datasets["val"]): views.val,
    }

    def objective(y_pred: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(dataset.label).ravel()

        view = dataset_to_view.get(id(dataset), views.train)
        if view.keys is None or "date" not in view.keys.columns:
            raise ValueError("pearsonr_ic_loss requires a 'date' column in view.keys.")

        dates = view.keys["date"].to_numpy()
        unique_dates = np.unique(dates)

        grad = np.zeros_like(y_pred, dtype=np.float64)
        hess = np.ones_like(y_pred, dtype=np.float64) * 1e-6
        n_days = max(1, len(unique_dates))

        for day in unique_dates:
            mask = dates == day
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue

            pred_day = y_pred[idx]
            true_day = y_true[idx]

            pred_mean = np.mean(pred_day)
            true_mean = np.mean(true_day)
            pred_centered = pred_day - pred_mean
            true_centered = true_day - true_mean

            pred_std = np.std(pred_day)
            true_std = np.std(true_day)

            if pred_std < 1e-8 or true_std < 1e-8:
                diff = pred_day - true_day
                grad[idx] = 2.0 * diff / n_days
                hess[idx] = 2.0
                continue

            cov_xy = np.mean(pred_centered * true_centered)
            grad_ic = (
                true_centered / (pred_std * true_std)
                - cov_xy * pred_centered / (pred_std ** 3 * true_std)
            ) / len(idx)

            grad[idx] = -grad_ic / n_days
            hess[idx] = np.abs(grad_ic) + 1e-6

        return grad, hess

    return objective


__all__ = ["FevalAdapter", "FevalAdapterFactory", "ObjectiveFactory"]
