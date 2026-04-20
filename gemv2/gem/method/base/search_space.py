"""
BaseSearchSpace -- abstract base class for hyper-parameter search spaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSearchSpace(ABC):
    """
    Search space ABC.

    Concrete implementations define which parameters to search and their ranges
    (e.g. LightGBMSearchSpace, XGBoostSearchSpace, MLPSearchSpace).
    """

    @abstractmethod
    def sample(self, trial: Any) -> Dict[str, Any]:
        """
        Sample one set of hyper-parameters from the search space.

        *trial* is an Optuna ``Trial`` object (or equivalent from another backend).
        """
        ...
