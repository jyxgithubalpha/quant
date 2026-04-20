"""
MetricRegistry -- pluggable metric system used by evaluators and feval callbacks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from scipy import stats


# =============================================================================
# Helpers
# =============================================================================


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "values"):
        return x.values
    return np.asarray(x)


def _extract_column(bundle: Any, column: str) -> np.ndarray:
    """Extract a column from bundle.keys or bundle.extra."""
    if hasattr(bundle, "keys") and bundle.keys is not None and column in bundle.keys.columns:
        return _to_numpy(bundle.keys[column])
    if hasattr(bundle, "extra") and bundle.extra is not None and column in bundle.extra.columns:
        return _to_numpy(bundle.extra[column])
    raise ValueError(f"Column '{column}' not found in keys/extra.")


def _extract_y(bundle: Any) -> np.ndarray:
    if not hasattr(bundle, "y"):
        raise ValueError("Bundle has no 'y' attribute.")
    return _to_numpy(bundle.y).ravel()


def _daily_pearson_ic(pred: np.ndarray, y_true: np.ndarray, dates: np.ndarray) -> np.ndarray:
    daily_ics = []
    for day in np.unique(dates):
        mask = dates == day
        if mask.sum() < 2:
            continue
        pred_day, y_day = pred[mask], y_true[mask]
        if np.std(pred_day) < 1e-8 or np.std(y_day) < 1e-8:
            continue
        ic, _ = stats.pearsonr(pred_day, y_day)
        if np.isfinite(ic):
            daily_ics.append(ic)
    return np.asarray(daily_ics, dtype=np.float64)


# =============================================================================
# Metric ABC + concrete metrics
# =============================================================================


class Metric(ABC):
    name: str = "base_metric"
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, pred: np.ndarray, bundle: Any) -> float:
        ...


class PearsonICMetric(Metric):
    name = "pearsonr_ic"
    higher_is_better = True

    def compute(self, pred, bundle):
        pred = np.asarray(pred).ravel()
        y_true = _extract_y(bundle)
        dates = _extract_column(bundle, "date")
        daily_ics = _daily_pearson_ic(pred, y_true, dates)
        return float(np.mean(daily_ics)) if daily_ics.size > 0 else 0.0


class ICIRMetric(Metric):
    name = "pearsonr_icir"
    higher_is_better = True

    def compute(self, pred, bundle):
        pred = np.asarray(pred).ravel()
        y_true = _extract_y(bundle)
        dates = _extract_column(bundle, "date")
        daily_ics = _daily_pearson_ic(pred, y_true, dates)
        if daily_ics.size == 0:
            return 0.0
        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics))
        return ic_mean / ic_std if ic_std > 1e-8 else 0.0


class MSEMetric(Metric):
    name = "mse"
    higher_is_better = False

    def compute(self, pred, bundle):
        pred = np.asarray(pred).ravel()
        y_true = _extract_y(bundle)
        return float(np.mean((pred - y_true) ** 2))


# =============================================================================
# Registry
# =============================================================================


class MetricRegistry:
    _registry: Dict[str, Metric] = {}

    @classmethod
    def register(cls, metric: Metric) -> None:
        cls._registry[metric.name] = metric

    @classmethod
    def get(cls, name: str) -> Metric:
        if name not in cls._registry:
            raise ValueError(
                f"Metric '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._registry


# Auto-register built-in metrics
MetricRegistry.register(PearsonICMetric())
MetricRegistry.register(ICIRMetric())
MetricRegistry.register(MSEMetric())
