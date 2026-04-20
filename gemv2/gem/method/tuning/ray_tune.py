"""
RayTuneTuner -- distributed hyper-parameter search via Ray Tune.
"""

import logging
from typing import Any, Dict, Optional

from ...core.data import ProcessedBundle
from ...core.training import TrainConfig, TuneResult
from ..base.search_space import BaseSearchSpace
from ..base.trainer import BaseTrainer
from ..base.tuner import BaseTuner

log = logging.getLogger(__name__)


class RayTuneTuner(BaseTuner):
    """
    Ray Tune-based distributed hyper-parameter search.

    Uses Optuna integration inside Ray Tune for efficient sampling.
    """

    def __init__(
        self,
        search_space: BaseSearchSpace,
        n_trials: int = 50,
        target_metric: str = "pearsonr_ic",
        direction: str = "maximize",
        num_workers: int = 4,
        resources_per_trial: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 42,
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.target_metric = target_metric
        self.direction = direction
        self.num_workers = num_workers
        self.resources_per_trial = resources_per_trial or {"cpu": 1, "gpu": 0}
        self.verbose = verbose
        self.seed = seed

    def tune(self, views: ProcessedBundle, trainer: BaseTrainer, config: TrainConfig) -> TuneResult:
        try:
            from ray import tune
            from ray.tune.search.optuna import OptunaSearch
        except ImportError as exc:
            raise ImportError("ray[tune] required: pip install 'ray[tune]'") from exc

        base_params = dict(config.params)
        mode = "max" if self.direction == "maximize" else "min"
        metric_key = "objective"

        ray_space = self._build_ray_space()

        optuna_search = OptunaSearch(metric=metric_key, mode=mode, seed=self.seed)

        def trainable(trial_config):
            merged = {**base_params, **trial_config}
            tc = config.for_tuning(merged, self.seed)
            fit_result = trainer.fit(views, tc, phase="tune")
            score = self._extract_metric(fit_result)
            return {metric_key: score}

        analysis = tune.run(
            trainable,
            config=ray_space,
            num_samples=self.n_trials,
            search_alg=optuna_search,
            resources_per_trial=self.resources_per_trial,
            verbose=self.verbose,
        )

        best_trial = analysis.get_best_trial(metric_key, mode)
        best = dict(best_trial.config)
        return TuneResult(
            best_params={**base_params, **best},
            best_value=float(best_trial.last_result[metric_key]),
            n_trials=len(analysis.trials),
        )

    def _build_ray_space(self) -> Dict[str, Any]:
        """Build Ray Tune config space from search_space if it supports it."""
        if hasattr(self.search_space, "to_ray_tune_space"):
            return self.search_space.to_ray_tune_space()
        raise NotImplementedError(
            f"{type(self.search_space).__name__} does not implement to_ray_tune_space(). "
            "Required for RayTuneTuner."
        )

    def _extract_metric(self, fit_result) -> float:
        for dataset_key in ("val", "valid", "validation"):
            dataset_metrics = fit_result.evals_result.get(dataset_key, {})
            scores = dataset_metrics.get(self.target_metric)
            if scores:
                best_iter = max(1, int(fit_result.best_iteration))
                idx = min(best_iter, len(scores)) - 1
                return float(scores[idx])
        return 0.0
