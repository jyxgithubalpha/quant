"""
Concrete state implementations: FeatureImportanceState, SampleWeightState, TuningState.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ...core.data import SplitView
from .base import BaseState


@dataclass
class FeatureImportanceState(BaseState):
    """
    Feature importance state

    Used for storing and updating EMA of feature importance, can be passed to FeatureWeightTransform.
    """
    importance_ema: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    feature_names_hash: Optional[str] = None
    alpha: float = 0.3

    @classmethod
    def state_key(cls) -> str:
        return "feature_importance"

    def update(
        self,
        new_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """Update feature importance EMA"""
        if new_importance is None:
            return

        # Update alpha
        if alpha is not None:
            self.alpha = alpha

        # Update feature names
        if feature_names is not None:
            self.feature_names = feature_names

        # Update EMA
        if self.importance_ema is None:
            self.importance_ema = new_importance.copy()
        else:
            self.importance_ema = self.alpha * new_importance + (1 - self.alpha) * self.importance_ema

    def to_transform_context(self) -> Dict[str, Any]:
        """Pass feature_weights to FeatureWeightTransform"""
        return {
            "feature_weights": self.importance_ema,
            "feature_names": self.feature_names,
        }

    def get_topk_indices(self, k: int) -> np.ndarray:
        """Get indices of top-k important features"""
        if self.importance_ema is None:
            return np.array([])
        k = min(k, len(self.importance_ema))
        return np.argsort(self.importance_ema)[-k:]


@dataclass
class SampleWeightState(BaseState):
    """
    Sample weight state

    Used for computing training sample weights, supports weighting by asset/time/industry.
    """
    asset_weights: Optional[Dict[str, float]] = None
    industry_weights: Optional[Dict[str, float]] = None
    time_weights: Optional[Dict[int, float]] = None

    @classmethod
    def state_key(cls) -> str:
        return "sample_weight"

    def update(
        self,
        asset_weights: Optional[Dict[str, float]] = None,
        industry_weights: Optional[Dict[str, float]] = None,
        time_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """Update sample weights"""
        if asset_weights is not None:
            self.asset_weights = asset_weights
        if industry_weights is not None:
            self.industry_weights = industry_weights
        if time_weights is not None:
            self.time_weights = time_weights

    def to_transform_context(self) -> Dict[str, Any]:
        return {
            "asset_weights": self.asset_weights,
            "industry_weights": self.industry_weights,
            "time_weights": self.time_weights,
        }

    def get_sample_weight(
        self,
        keys: pl.DataFrame,
        group: Optional[pl.DataFrame] = None,
        industry_col: str = "industry",
    ) -> np.ndarray:
        """Compute sample weights"""
        n = keys.height
        weights = np.ones(n, dtype=np.float32)

        if self.asset_weights is not None:
            codes = keys["code"].to_numpy()
            for i, code in enumerate(codes):
                if code in self.asset_weights:
                    weights[i] *= self.asset_weights[code]

        if self.time_weights is not None:
            dates = keys["date"].to_numpy()
            for i, d in enumerate(dates):
                if d in self.time_weights:
                    weights[i] *= self.time_weights[d]

        if self.industry_weights is not None and group is not None:
            if industry_col in group.columns:
                industries = group[industry_col].to_numpy()
                for i, ind in enumerate(industries):
                    if ind in self.industry_weights:
                        weights[i] *= self.industry_weights[ind]

        return weights

    def get_weights_for_view(
        self,
        view: SplitView,
        industry_col: str = "industry",
    ) -> np.ndarray:
        return self.get_sample_weight(
            keys=view.keys,
            group=view.group if view.group is not None else view.extra,
            industry_col=industry_col,
        )


@dataclass
class TuningState(BaseState):
    """
    Tuning state - for hyperparameter search optimization
    """
    last_best_params: Optional[Dict[str, Any]] = None
    params_history: List[Dict[str, Any]] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    search_space_shrink_ratio: float = 0.5

    @classmethod
    def state_key(cls) -> str:
        return "tuning"

    def update(self, best_params: Dict[str, Any], best_objective: float) -> None:
        """Update tuning state"""
        self.last_best_params = best_params.copy()
        self.params_history.append(best_params.copy())
        self.objective_history.append(best_objective)

    def to_transform_context(self) -> Dict[str, Any]:
        return {
            "last_best_params": self.last_best_params,
        }

    def get_shrunk_space(
        self,
        base_space: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Shrink search space based on history"""
        if self.last_best_params is None:
            return base_space

        shrunk_space = {}
        ratio = self.search_space_shrink_ratio

        for name, (low, high) in base_space.items():
            if name in self.last_best_params:
                center = self.last_best_params[name]
                half_range = (high - low) * ratio / 2
                new_low = max(low, center - half_range)
                new_high = min(high, center + half_range)
                shrunk_space[name] = (new_low, new_high)
            else:
                shrunk_space[name] = (low, high)

        return shrunk_space
