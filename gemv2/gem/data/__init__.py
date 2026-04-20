"""
Data module (new) - uses core/ types, renamed classes.
"""

from .module import DataModule
from .reader import DataReader, FeatherReader
from .preprocessors import (
    SingleSourceDataPreprocessor,
    SourcePipeline,
    SingleSourceDataPreprocessorPipeline,  # backwards-compatible alias
    MultiSourceDataPreprocessor,
    CrossSourcePipeline,
    MultiSourceDataPreprocessorPipeline,   # backwards-compatible alias
    AlignPreprocessor,
    DropDuplicatesPreprocessor,
    DropNaNPreprocessor,
    FillNaNPreprocessor,
    ColumnFilterPreprocessor,
    CodeFilterPreprocessor,
    DateFilterPreprocessor,
    RenameColumnsPreprocessor,
    DTypePreprocessor,
)
from .assembler import DatasetAssembler, GlobalDataAssembler, FeatureAssembler
from .split_generator import (
    SplitGenerator,
    RollingWindowSplitGenerator,
    TimeSeriesKFoldSplitGenerator,
)

# Core types re-exported for convenience
from ..core.data import (
    SourceSpec,
    DatasetSpec,
    GlobalDataset,
    SplitView,
    SplitBundle,
    ProcessedBundle,
    SplitSpec,
    SplitPlan,
)

__all__ = [
    # Module
    "DataModule",
    # Readers
    "DataReader",
    "FeatherReader",
    # Single-source preprocessors
    "SingleSourceDataPreprocessor",
    "SourcePipeline",
    "SingleSourceDataPreprocessorPipeline",
    # Multi-source preprocessors
    "MultiSourceDataPreprocessor",
    "CrossSourcePipeline",
    "MultiSourceDataPreprocessorPipeline",
    # Concrete preprocessors
    "AlignPreprocessor",
    "DropDuplicatesPreprocessor",
    "DropNaNPreprocessor",
    "FillNaNPreprocessor",
    "ColumnFilterPreprocessor",
    "CodeFilterPreprocessor",
    "DateFilterPreprocessor",
    "RenameColumnsPreprocessor",
    "DTypePreprocessor",
    # Assemblers
    "DatasetAssembler",
    "GlobalDataAssembler",
    "FeatureAssembler",
    # Split generators
    "SplitGenerator",
    "RollingWindowSplitGenerator",
    "TimeSeriesKFoldSplitGenerator",
    # Core types
    "SourceSpec",
    "DatasetSpec",
    "GlobalDataset",
    "SplitView",
    "SplitBundle",
    "ProcessedBundle",
    "SplitSpec",
    "SplitPlan",
]
