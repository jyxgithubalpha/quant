"""
MLP sub-package for torch method.
"""

from .model import FactorMLP
from .search_space import MLPSearchSpace

__all__ = ["FactorMLP", "MLPSearchSpace"]
