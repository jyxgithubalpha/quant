"""
LightGBM hyperparameter search space.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...base.search_space import BaseSearchSpace


@dataclass
class LightGBMSearchSpace(BaseSearchSpace):
    """LightGBM hyperparameter search space."""

    learning_rate: Tuple[float, float] = (0.01, 0.3)
    num_leaves: Tuple[int, int] = (20, 300)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_samples: Tuple[int, int] = (5, 100)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)

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
        nl = space.get("num_leaves", self.num_leaves)
        md = space.get("max_depth", self.max_depth)
        mcs = space.get("min_child_samples", self.min_child_samples)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)

        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "num_leaves": trial.suggest_int("num_leaves", int(nl[0]), int(nl[1])),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_child_samples": trial.suggest_int("min_child_samples", int(mcs[0]), int(mcs[1])),
            "subsample": trial.suggest_float("subsample", *ss),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl, log=True),
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
        nl = space.get("num_leaves", self.num_leaves)
        md = space.get("max_depth", self.max_depth)
        mcs = space.get("min_child_samples", self.min_child_samples)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)

        return {
            "learning_rate": tune.loguniform(*lr),
            "num_leaves": tune.randint(int(nl[0]), int(nl[1]) + 1),
            "max_depth": tune.randint(int(md[0]), int(md[1]) + 1),
            "min_child_samples": tune.randint(int(mcs[0]), int(mcs[1]) + 1),
            "subsample": tune.uniform(*ss),
            "colsample_bytree": tune.uniform(*cs),
            "reg_alpha": tune.loguniform(*ra),
            "reg_lambda": tune.loguniform(*rl),
        }


__all__ = ["LightGBMSearchSpace"]
