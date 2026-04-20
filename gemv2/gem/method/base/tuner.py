"""
BaseTuner -- abstract base class for hyper-parameter / architecture search.
"""

from abc import ABC, abstractmethod

from ...core.data import ProcessedBundle
from ...core.training import TrainConfig, TuneResult
from .trainer import BaseTrainer


class BaseTuner(ABC):
    """
    Tuner ABC.

    Concrete implementations:
    - OptunaTuner:   trial-based HP search via Optuna.
    - RayTuneTuner:  distributed HP search via Ray Tune.
    - NNITuner:      neural architecture search via NNI (weight-sharing).

    The returned ``TuneResult.best_params`` may contain:
    - scalar hyper-parameters (HP search), or
    - an ``"__arch__"`` key with a network topology description (NAS).
    """

    @abstractmethod
    def tune(
        self,
        views: ProcessedBundle,
        trainer: BaseTrainer,
        config: TrainConfig,
    ) -> TuneResult:
        """
        Search for the best configuration.

        Args:
            views: processed train/val/test data.
            trainer: used by trial-based tuners to train each trial.
            config: baseline training configuration.

        Returns:
            TuneResult with best_params and best metric value.
        """
        ...
