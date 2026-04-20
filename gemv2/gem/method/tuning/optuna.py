"""
OptunaTuner -- trial-based hyper-parameter search via Optuna.
"""

import logging
from typing import Any, Dict, Optional

from ...core.data import ProcessedBundle
from ...core.training import TrainConfig, TuneResult
from ..base.search_space import BaseSearchSpace
from ..base.trainer import BaseTrainer
from ..base.tuner import BaseTuner

log = logging.getLogger(__name__)


class OptunaTuner(BaseTuner):
    """
    Optuna-based hyper-parameter search.

    For each trial:  sample params → trainer.fit(phase="tune") → extract metric → report.
    """

    def __init__(
        self,
        search_space: BaseSearchSpace,
        n_trials: int = 50,
        target_metric: str = "pearsonr_ic",
        direction: str = "maximize",
        seed: int = 42,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.target_metric = target_metric
        self.direction = direction
        self.seed = seed
        self.n_jobs = n_jobs
        self.timeout = timeout

    def tune(self, views: ProcessedBundle, trainer: BaseTrainer, config: TrainConfig) -> TuneResult:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        base_params = dict(config.params)

        def objective(trial) -> float:
            sampled = self.search_space.sample(trial)
            merged = {**base_params, **sampled}
            trial_config = config.for_tuning(merged, self.seed)
            fit_result = trainer.fit(views, trial_config, phase="tune")
            return self._extract_metric(fit_result)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(
            objective, n_trials=self.n_trials,
            n_jobs=self.n_jobs, timeout=self.timeout,
        )

        best = dict(study.best_trial.params)
        return TuneResult(
            best_params={**base_params, **best},
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            all_trials=[
                {"params": t.params, "value": t.value, "state": str(t.state)}
                for t in study.trials
            ],
        )

    def _extract_metric(self, fit_result) -> float:
        """Extract target_metric from trainer's evals_result."""
        for dataset_key in ("val", "valid", "validation"):
            dataset_metrics = fit_result.evals_result.get(dataset_key, {})
            scores = dataset_metrics.get(self.target_metric)
            if scores:
                best_iter = max(1, int(fit_result.best_iteration))
                idx = min(best_iter, len(scores)) - 1
                return float(scores[idx])
        log.warning(
            "Metric '%s' not found in val evals_result, returning 0.0. "
            "Available keys: %s", self.target_metric, list(fit_result.evals_result.keys()),
        )
        return 0.0
