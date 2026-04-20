"""
Data transforms -- feature / label processing pipeline.

All transforms follow the fit(train) -> transform(all) pattern.
"""

from .base import BaseTransform, extract_date_keys
from .feature_weight import FeatureWeighter
from .fillnan import FillNaN
from .mad_standardize import MADStandardize
from .min_sample_filter import MinSampleFilter
from .pipeline import TransformPipeline
from .rank import RankNormalize
from .spectral import SpectralCluster
from .standardize import ZScoreStandardize
from .stats import StatsCalculator, TransformStats
from .winsorize import Winsorize

__all__ = [
    # base
    "BaseTransform",
    "TransformPipeline",
    "extract_date_keys",
    # concrete transforms
    "MADStandardize",
    "Winsorize",
    "RankNormalize",
    "ZScoreStandardize",
    "FillNaN",
    "MinSampleFilter",
    "FeatureWeighter",
    "SpectralCluster",
    # stats
    "StatsCalculator",
    "TransformStats",
]
