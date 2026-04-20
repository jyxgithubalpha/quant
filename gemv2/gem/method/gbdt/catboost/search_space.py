"""
CatBoost hyperparameter search space.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...base.search_space import BaseSearchSpace


@dataclass
class CatBoostSearchSpace(BaseSearchSpace):
    """CatBoost hyperparameter search space."""

    learning_rate: Tuple[float, float] = (0.01, 0.3)
    depth: Tuple[int, int] = (4, 10)
    l2_leaf_reg: Tuple[float, float] = (1.0, 10.0)
    bagging_temperature: Tuple[float, float] = (0.0, 1.0)
    random_strength: Tuple[float, float] = (0.0, 10.0)
    border_count: Tuple[int, int] = (32, 255)

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
        dp = space.get("depth", self.depth)
        l2 = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt = space.get("bagging_temperature", self.bagging_temperature)
        rs = space.get("random_strength", self.random_strength)
        bc = space.get("border_count", self.border_count)

        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "depth": trial.suggest_int("depth", int(dp[0]), int(dp[1])),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *l2),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *bt),
            "random_strength": trial.suggest_float("random_strength", *rs),
            "border_count": trial.suggest_int("border_count", int(bc[0]), int(bc[1])),
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
        dp = space.get("depth", self.depth)
        l2 = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt = space.get("bagging_temperature", self.bagging_temperature)
        rs = space.get("random_strength", self.random_strength)
        bc = space.get("border_count", self.border_count)

        return {
            "learning_rate": tune.loguniform(*lr),
            "depth": tune.randint(int(dp[0]), int(dp[1]) + 1),
            "l2_leaf_reg": tune.uniform(*l2),
            "bagging_temperature": tune.uniform(*bt),
            "random_strength": tune.uniform(*rs),
            "border_count": tune.randint(int(bc[0]), int(bc[1]) + 1),
        }


__all__ = ["CatBoostSearchSpace"]
