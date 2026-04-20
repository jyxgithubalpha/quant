"""
BaseAdapter -- abstract base class for dataset format conversion.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ...core.data import SplitView


class BaseAdapter(ABC):
    """
    Dataset adapter ABC.

    Converts a SplitView to a framework-specific dataset object
    (e.g. ``lgb.Dataset``, ``xgb.DMatrix``, ``catboost.Pool``).
    """

    @abstractmethod
    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Convert a SplitView to a backend dataset.

        Args:
            view: SplitView with X, y, keys.
            reference: reference dataset (for val/test sets that need train stats).
            **kwargs: backend-specific options.

        Returns:
            Backend dataset instance.
        """
        ...
