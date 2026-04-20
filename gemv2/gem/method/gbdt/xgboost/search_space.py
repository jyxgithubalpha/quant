"""
XGBoost hyperparameter search space.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...base.search_space import BaseSearchSpace


@dataclass
class XGBoostSearchSpace(BaseSearchSpace):
    """XGBoost hyperparameter search space."""

    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_weight: Tuple[int, int] = (1, 10)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)
    gamma: Tuple[float, float] = (0.0, 5.0)

    def sample(self, trial: Any) -> Dict[str, Any]:
        """Sample one set of hyper-parameters (delegates to sample_optuna)."""
        return self.sample_optuna(trial)

    def sample_optuna(
        self,
        trial: Any,
        shrunk_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        space = shrunk_space or {}

        lr = space.get("learning_rate", self.learning_rate)
        md = space.get("max_depth", self.max_depth)
        mcw = space.get("min_child_weight", self.min_child_weight)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)
        gm = space.get("gamma", self.gamma)

        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_child_weight": trial.suggest_int("min_child_weight", int(mcw[0]), int(mcw[1])),
            "subsample": trial.suggest_float("subsample", *ss),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl, log=True),
            "gamma": trial.suggest_float("gamma", *gm),
        }

    def to_ray_tune_space(
        self,
        shrunk_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            from ray import tune
        except ImportError as exc:
            raise ImportError(
                "ray[tune] is required. Install with: pip install 'ray[tune]'"
            ) from exc

        space = shrunk_space or {}

        lr = space.get("learning_rate", self.learning_rate)
        md = space.get("max_depth", self.max_depth)
        mcw = space.get("min_child_weight", self.min_child_weight)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)
        gm = space.get("gamma", self.gamma)

        return {
            "learning_rate": tune.loguniform(*lr),
            "max_depth": tune.randint(int(md[0]), int(md[1]) + 1),
            "min_child_weight": tune.randint(int(mcw[0]), int(mcw[1]) + 1),
            "subsample": tune.uniform(*ss),
            "colsample_bytree": tune.uniform(*cs),
            "reg_alpha": tune.loguniform(*ra),
            "reg_lambda": tune.loguniform(*rl),
            "gamma": tune.uniform(*gm),
        }


__all__ = ["XGBoostSearchSpace"]
