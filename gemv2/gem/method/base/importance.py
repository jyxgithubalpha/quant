"""
BaseImportanceExtractor -- abstract base class for feature importance extraction.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np
import polars as pl


class BaseImportanceExtractor(ABC):
    """
    Feature importance extractor ABC.

    Concrete implementations:
    - GBDTImportanceExtractor:     gain / split importance from tree models.
    - GradientImportanceExtractor: gradient-based importance for neural nets.
    - SklearnImportanceExtractor:  coef_ based importance for linear models.
    """

    @abstractmethod
    def extract(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, Optional[pl.DataFrame]]:
        """
        Extract feature importance.

        Args:
            model: trained model.
            feature_names: feature name list (length must match output vector).

        Returns:
            (importance_vector, importance_dataframe).
            Vector shape = (n_features,).
        """
        ...
