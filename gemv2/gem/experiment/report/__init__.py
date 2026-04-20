"""Report subpackage."""

from .config import VisualizationConfig

__all__ = [
    "VisualizationConfig",
]


def __getattr__(name):
    """Lazy import ReportGenerator to break circular import with configs.py."""
    if name == "ReportGenerator":
        from .generator import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
