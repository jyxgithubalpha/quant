"""
Visualization configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VisualizationRenderConfig:
    show: bool = False
    interval_ms: int = 800
    dpi: int = 150


@dataclass
class ImportanceVizConfig:
    enabled: bool = True
    export_data: bool = True
    heatmap: bool = True
    animation: bool = True
    distribution: bool = True
    sort_by: str = "mean"
    normalize: str = "zscore"


@dataclass
class MetricsVizConfig:
    enabled: bool = True
    export_data: bool = True
    quarterly_summary: bool = True
    cumulative_backtest: bool = True
    relative_improvement_curve: bool = True
    overview: bool = True
    distribution: bool = True
    per_metric: bool = True
    metric_names: Optional[List[str]] = None


@dataclass
class VisualizationConfig:
    enabled: bool = False
    output_subdir: str = "visualization"
    render: VisualizationRenderConfig = field(default_factory=VisualizationRenderConfig)
    importance: ImportanceVizConfig = field(default_factory=ImportanceVizConfig)
    metrics: MetricsVizConfig = field(default_factory=MetricsVizConfig)
