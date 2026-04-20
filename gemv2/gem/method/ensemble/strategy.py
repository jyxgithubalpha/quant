"""
BaseEnsembleStrategy and simple built-in strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ...core.data import SplitView


class BaseEnsembleStrategy(ABC):
    """
    Ensemble strategy ABC.

    fit()    -- learn combination weights on validation predictions.
    combine() -- merge multiple model predictions into one array.
    """

    @abstractmethod
    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        view: SplitView,
    ) -> "BaseEnsembleStrategy":
        """
        Learn ensemble parameters from validation predictions.

        Args:
            predictions: ``{model_name: val_prediction_array}``.
            view: validation SplitView (has y, keys, extra).
        """
        ...

    @abstractmethod
    def combine(
        self,
        predictions: Dict[str, np.ndarray],
        view: SplitView,
    ) -> np.ndarray:
        """
        Combine multiple model predictions into one.

        Args:
            predictions: ``{model_name: prediction_array}`` for any split.
            view: the corresponding SplitView.

        Returns:
            Combined prediction array.
        """
        ...


class MeanEnsemble(BaseEnsembleStrategy):
    """Simple equal-weight average."""

    def fit(self, predictions, view):
        return self

    def combine(self, predictions, view):
        return np.mean(list(predictions.values()), axis=0)


class ICWeightedEnsemble(BaseEnsembleStrategy):
    """Weight each model by its validation IC (Pearson correlation with y)."""

    def __init__(self) -> None:
        self._weights: Dict[str, float] = {}

    def fit(self, predictions, view):
        y = view.y.ravel()
        raw_weights: Dict[str, float] = {}
        for name, pred in predictions.items():
            pred_flat = pred.ravel()
            if len(pred_flat) < 2 or np.std(pred_flat) < 1e-8 or np.std(y) < 1e-8:
                raw_weights[name] = 0.0
            else:
                ic = float(np.corrcoef(pred_flat, y)[0, 1])
                raw_weights[name] = max(ic, 0.0) if np.isfinite(ic) else 0.0

        total = sum(raw_weights.values()) or 1.0
        self._weights = {k: v / total for k, v in raw_weights.items()}
        return self

    def combine(self, predictions, view):
        result = np.zeros_like(next(iter(predictions.values())), dtype=np.float64)
        for name, pred in predictions.items():
            result += pred * self._weights.get(name, 0.0)
        return result
