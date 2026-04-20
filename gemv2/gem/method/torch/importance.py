"""
Gradient-based feature importance extractor for PyTorch tabular models.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import polars as pl

from ..base.importance import BaseImportanceExtractor


class GradientImportanceExtractor(BaseImportanceExtractor):
    """
    Feature importance via first-layer weight norms (proxy for gradient magnitude).

    Falls back to uniform importance when the model does not expose
    ``feature_importance()``.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def extract(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, Optional[pl.DataFrame]]:
        n_features = len(feature_names)
        base = model.model if hasattr(model, "model") else model
        importance = self._extract_from_model(base, n_features)

        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)

        df = pl.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort("importance", descending=True)
        return importance, df

    @staticmethod
    def _extract_from_model(model: Any, n_features: int) -> np.ndarray:
        if hasattr(model, "feature_importance"):
            raw = model.feature_importance().detach().cpu().numpy().astype(np.float32)
            if raw.shape[0] == n_features:
                return raw
        return np.ones(n_features, dtype=np.float32)
