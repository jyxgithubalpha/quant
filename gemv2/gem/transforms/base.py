"""
BaseTransform -- abstract base class for all data transforms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..core.training import TransformContext


def extract_date_keys(
    keys: Optional[pl.DataFrame],
    date_col: str = "date",
) -> Optional[np.ndarray]:
    """Extract date column from keys DataFrame as numpy array."""
    if keys is None or date_col not in keys.columns:
        return None
    return keys[date_col].to_numpy()


class BaseTransform(ABC):
    """
    Transform base class.

    Supports receiving external parameters through *context*
    (e.g. feature weights from RollingState).
    Subclasses access context via ``self._context``.
    """

    def __init__(self) -> None:
        self._context: TransformContext = {}

    def set_context(self, ctx: TransformContext) -> "BaseTransform":
        """Set context parameters (called by TransformPipeline)."""
        self._context = ctx
        return self

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self._context.get(key, default)

    # -- core interface ---------------------------------------------------

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> "BaseTransform":
        """Fit on training data.  *keys* is date array for cross-sectional grouping."""
        ...

    @abstractmethod
    def transform(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transform.  Returns ``(X_transformed, y_transformed)``."""
        ...

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one call."""
        self.fit(X, y, keys)
        return self.transform(X, y, keys)

    def inverse_transform(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform (optional, default is identity)."""
        return X, y

    # -- feature name tracking --------------------------------------------

    def get_output_feature_names(self, input_names: List[str]) -> List[str]:
        """
        Given input feature names, return output feature names.

        Override in transforms that change the number of columns
        (e.g. SpectralCluster adds cols, FeatureWeighter may remove cols).
        Default: identity (same names in, same names out).
        """
        return input_names
