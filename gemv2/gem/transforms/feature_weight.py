"""
FeatureWeighter -- weight or select features based on importance from context.
"""

from typing import List, Literal, Optional, Tuple

import numpy as np

from .base import BaseTransform


class FeatureWeighter(BaseTransform):
    """
    Feature weighting / selection transform.

    Reads ``feature_weights`` from context (injected by RollingState)
    and applies one of several weighting strategies.

    Methods:
        multiply:       X * weights
        sqrt_multiply:  X * sqrt(weights)
        rank_weight:    X * rank_normalized_weights
        softmax:        X * softmax(weights / temperature)
        select_topk:    keep only top-k important features
    """

    def __init__(
        self,
        method: Literal["multiply", "sqrt_multiply", "rank_weight", "softmax", "select_topk"] = "multiply",
        topk: Optional[int] = None,
        temperature: float = 1.0,
        min_weight: float = 0.0,
        normalize: bool = False,
        fallback: Literal["uniform", "ones", "skip"] = "ones",
        context_key: str = "feature_weights",
    ) -> None:
        super().__init__()
        self.method = method
        self.topk = topk
        self.temperature = temperature
        self.min_weight = min_weight
        self.normalize = normalize
        self.fallback = fallback
        self.context_key = context_key
        self._weights: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None

    def fit(self, X, y, keys=None):
        n_features = X.shape[1]
        raw_weights = self.get_context_value(self.context_key)

        if raw_weights is None:
            raw_weights = self._fallback_weights(n_features)
        else:
            raw_weights = np.asarray(raw_weights, dtype=np.float32)

        if self.method == "multiply":
            self._weights = self._constrain(raw_weights, n_features)
        elif self.method == "sqrt_multiply":
            self._weights = self._constrain(np.sqrt(np.maximum(raw_weights, 0)), n_features)
        elif self.method == "rank_weight":
            ranks = np.argsort(np.argsort(raw_weights)).astype(np.float32)
            self._weights = self._constrain((ranks + 1) / n_features, n_features)
        elif self.method == "softmax":
            exp_w = np.exp((raw_weights - raw_weights.max()) / self.temperature)
            self._weights = self._constrain(exp_w / exp_w.sum() * n_features, n_features)
        elif self.method == "select_topk":
            k = min(self.topk or n_features, n_features)
            self._selected_indices = np.sort(np.argsort(raw_weights)[-k:])
            self._weights = None
        return self

    def transform(self, X, y, keys=None):
        if self.fallback == "skip" and self._weights is None and self._selected_indices is None:
            return X, y

        X_out = X.copy()
        if self.method == "select_topk" and self._selected_indices is not None:
            X_out = X_out[:, self._selected_indices]
        elif self._weights is not None:
            X_out = X_out * self._weights
        return X_out, y

    def get_output_feature_names(self, input_names: List[str]) -> List[str]:
        if self.method == "select_topk" and self._selected_indices is not None:
            return [input_names[i] for i in self._selected_indices]
        return input_names

    # -- helpers ----------------------------------------------------------

    def _fallback_weights(self, n: int) -> np.ndarray:
        if self.fallback == "uniform":
            return np.full(n, 1.0 / n, dtype=np.float32)
        return np.ones(n, dtype=np.float32)

    def _constrain(self, w: np.ndarray, n: int) -> np.ndarray:
        w = w.copy()
        if self.min_weight > 0:
            w = np.maximum(w, self.min_weight)
        if self.normalize and w.sum() > 0:
            w = w * n / w.sum()
        return w
