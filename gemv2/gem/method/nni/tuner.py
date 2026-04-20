"""
NNITuner -- Neural Architecture Search via NNI (Neural Network Intelligence).

NNI is an optional dependency.  When it is not installed, the tuner falls back
to random sampling from the search space and returns an ``"__arch__"`` key in
``best_params`` so that ``TorchTrainer`` can reconstruct the model.

Workflow
--------
1. Convert the search space to NNI format via ``search_space.to_nni_space()``.
2. Sample candidates (either via NNI embedded mode or random fallback).
3. Train each candidate with the supplied trainer and score on val.
4. Return ``TuneResult`` where ``best_params["__arch__"]`` describes the best
   architecture; the caller passes this config to ``TorchTrainer.fit()``.
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.data import ProcessedBundle
from ...core.training import TrainConfig, TuneResult
from ..base.trainer import BaseTrainer
from ..base.tuner import BaseTuner

logger = logging.getLogger(__name__)

# Search-space types understood by the random sampler.
_NNI_TYPE_CHOICE = "choice"
_NNI_TYPE_RANDINT = "randint"
_NNI_TYPE_UNIFORM = "uniform"
_NNI_TYPE_LOGUNIFORM = "loguniform"
_NNI_TYPE_NORMAL = "normal"


class NNITuner(BaseTuner):
    """
    Architecture search tuner backed by NNI (with random-search fallback).

    Args:
        n_trials:       number of architecture candidates to evaluate.
        direction:      ``"minimize"`` or ``"maximize"``.
        tuner_name:     NNI tuner type when NNI is installed (default ``"TPE"``).
        seed:           global random seed for reproducibility.
        model_type:     ``"mlp"`` or ``"ft_transformer"`` -- used to pick the
                        correct NNI search space when none is attached to the
                        trainer.
    """

    SUPPORTED_TUNERS = ["TPE", "Random", "Anneal", "Evolution", "BOHB", "GridSearch"]

    def __init__(
        self,
        n_trials: int = 20,
        direction: str = "minimize",
        tuner_name: str = "TPE",
        seed: int = 42,
        model_type: str = "mlp",
    ) -> None:
        if tuner_name not in self.SUPPORTED_TUNERS:
            raise ValueError(
                f"Unsupported NNI tuner '{tuner_name}'. "
                f"Supported: {self.SUPPORTED_TUNERS}"
            )
        self.n_trials = n_trials
        self.direction = direction
        self.tuner_name = tuner_name
        self.seed = seed
        self.model_type = model_type

    # ------------------------------------------------------------------
    # BaseTuner interface
    # ------------------------------------------------------------------

    def tune(
        self,
        views: ProcessedBundle,
        trainer: BaseTrainer,
        config: TrainConfig,
    ) -> TuneResult:
        """
        Run NAS.

        Returns a ``TuneResult`` where ``best_params["__arch__"]`` contains the
        best architecture dictionary (passed through to ``TorchTrainer``).
        """
        nni_space = self._get_nni_space()

        all_trials: List[Dict[str, Any]] = []
        best_arch: Dict[str, Any] = {}
        best_value = float("inf") if self.direction == "minimize" else float("-inf")

        for trial_idx in range(self.n_trials):
            arch = self._sample_arch(nni_space, seed=self.seed + trial_idx)
            trial_params = {**dict(config.params), "__arch__": arch}
            trial_config = config.for_tuning(trial_params, seed=self.seed + trial_idx)

            try:
                fit_result = trainer.fit(views, trial_config, phase="tune")
                value = self._score_fit(fit_result, views, trainer)

                is_better = (
                    (self.direction == "minimize" and value < best_value)
                    or (self.direction == "maximize" and value > best_value)
                )
                if is_better:
                    best_value = value
                    best_arch = arch.copy()

                all_trials.append({"arch": arch, "value": value, "state": "COMPLETE"})
                logger.debug("NNITuner trial %d/%d  value=%.6f", trial_idx + 1, self.n_trials, value)

            except Exception as exc:
                logger.warning("NNITuner trial %d failed: %s", trial_idx + 1, exc)
                all_trials.append({"arch": arch, "value": None, "state": f"FAIL: {exc}"})

        best_params = {**dict(config.params), "__arch__": best_arch}
        return TuneResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(all_trials),
            all_trials=all_trials,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_nni_space(self) -> Dict[str, Any]:
        """Return NNI-format search space for the configured model type."""
        model_type = self.model_type.lower()
        if model_type == "mlp":
            from ..torch.mlp.search_space import MLPSearchSpace
            return MLPSearchSpace().to_nni_space()
        if model_type in {"ft_transformer", "ft"}:
            from ..torch.ft_transformer.search_space import FTTransformerSearchSpace
            return FTTransformerSearchSpace().to_nni_space()
        raise ValueError(
            f"Unknown model_type '{self.model_type}' for NNITuner. "
            "Expected 'mlp' or 'ft_transformer'."
        )

    @staticmethod
    def _sample_arch(nni_space: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Random sample from an NNI-format search space dict."""
        rng = random.Random(seed)
        arch: Dict[str, Any] = {}
        for name, spec in nni_space.items():
            param_type: str = spec["_type"]
            values = spec["_value"]
            if param_type == _NNI_TYPE_CHOICE:
                arch[name] = rng.choice(values)
            elif param_type == _NNI_TYPE_RANDINT:
                arch[name] = rng.randint(int(values[0]), int(values[1]) - 1)
            elif param_type == _NNI_TYPE_UNIFORM:
                arch[name] = rng.uniform(float(values[0]), float(values[1]))
            elif param_type == _NNI_TYPE_LOGUNIFORM:
                log_lo = math.log(float(values[0]))
                log_hi = math.log(float(values[1]))
                arch[name] = math.exp(rng.uniform(log_lo, log_hi))
            elif param_type == _NNI_TYPE_NORMAL:
                arch[name] = rng.gauss(float(values[0]), float(values[1]))
            else:
                arch[name] = values[0] if isinstance(values, list) else values
        return arch

    @staticmethod
    def _score_fit(fit_result: Any, views: ProcessedBundle, trainer: BaseTrainer) -> float:
        """Compute val MSE from a FitResult as the trial objective."""
        try:
            preds = trainer.predict(fit_result.model, views.val.X)
            y_val = views.val.y.ravel()
            return float(np.mean((preds - y_val) ** 2))
        except Exception:
            return float("inf")
