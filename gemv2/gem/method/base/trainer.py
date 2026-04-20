"""
BaseTrainer -- abstract base class for all model trainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ...core.data import ProcessedBundle
from ...core.training import FitResult, TrainConfig


class BaseTrainer(ABC):
    """
    Trainer ABC.

    Subclasses implement framework-specific training (LightGBM, XGBoost, PyTorch, ...)
    and prediction logic.
    """

    @abstractmethod
    def fit(
        self,
        views: ProcessedBundle,
        config: TrainConfig,
        phase: str = "full",
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> FitResult:
        """
        Train a model.

        Args:
            views: processed train/val/test data.
            config: training hyper-parameters and iteration limits.
            phase: ``"full"`` for final training, ``"tune"`` for lightweight trial.
            sample_weights: optional per-split weight arrays ``{"train": ..., ...}``.

        Returns:
            FitResult containing the trained model and metadata.
        """
        ...

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from a trained model.

        Each trainer knows how to call its own model type
        (e.g. ``lgb.Booster.predict``, ``xgb.Booster.inplace_predict``, ``model(tensor)``).
        """
        ...
