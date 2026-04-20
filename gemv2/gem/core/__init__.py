"""
Core type definitions shared across all gem layers.

This module has ZERO internal gem dependencies -- only numpy and polars.
"""

from .data import (
    GlobalDataset,
    ProcessedBundle,
    SplitBundle,
    SplitPlan,
    SplitSpec,
    SplitView,
)
from .training import (
    EvalResult,
    FitResult,
    RunOutput,
    StateDelta,
    TrainConfig,
    TransformContext,
    TuneResult,
)

__all__ = [
    # data containers
    "SplitSpec",
    "SplitPlan",
    "GlobalDataset",
    "SplitView",
    "SplitBundle",
    "ProcessedBundle",
    # training types
    "TransformContext",
    "TrainConfig",
    "TuneResult",
    "FitResult",
    "EvalResult",
    "StateDelta",
    "RunOutput",
]
