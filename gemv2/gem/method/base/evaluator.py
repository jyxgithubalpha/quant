"""
BaseEvaluator -- abstract base class for prediction evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ...core.data import ProcessedBundle
from ...core.training import EvalResult


class BaseEvaluator(ABC):
    """
    Evaluator ABC.

    Receives pre-computed predictions (not the model itself) and computes
    metrics for each split.
    """

    @abstractmethod
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        views: ProcessedBundle,
    ) -> Dict[str, EvalResult]:
        """
        Evaluate predictions.

        Args:
            predictions: ``{"train": array, "val": array, "test": array}``.
            views: the processed views that produced these predictions.

        Returns:
            ``{"train": EvalResult, "val": EvalResult, "test": EvalResult}``.
        """
        ...
