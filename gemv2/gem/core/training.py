"""
Core training / evaluation / output types.

All types here depend ONLY on numpy, polars, pathlib -- no gem internal imports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl


# =============================================================================
# Shared type alias
# =============================================================================

TransformContext = Dict[str, Any]
"""Context passed from RollingState to TransformPipeline (feature weights, etc.)."""


# =============================================================================
# Training configuration
# =============================================================================


@dataclass
class TrainConfig:
    """
    Framework-agnostic training configuration.

    Attributes:
        params:  model hyper-parameters (framework-specific keys inside).
        max_iterations:  rounds (GBDT) or epochs (torch).
        early_stopping_patience:  stop after N iterations without improvement.
        monitor_metrics:  metrics tracked during training for early-stop decisions.
        seed:  random seed.
        log_interval:  print training log every N iterations (0 = silent).
    """

    params: Dict[str, Any]
    max_iterations: int = 1000
    early_stopping_patience: int = 50
    monitor_metrics: List[str] = field(default_factory=lambda: ["pearsonr_ic"])
    seed: int = 42
    log_interval: int = 100

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be > 0, got {self.max_iterations}"
            )
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be > 0, got {self.early_stopping_patience}"
            )
        if not self.monitor_metrics:
            raise ValueError("monitor_metrics must not be empty.")

    def for_tuning(self, trial_params: Dict[str, Any], seed: int) -> "TrainConfig":
        """Create a lightweight config for one tuning trial."""
        return TrainConfig(
            params=trial_params,
            max_iterations=self.max_iterations,
            early_stopping_patience=self.early_stopping_patience,
            monitor_metrics=self.monitor_metrics[:1],
            seed=seed,
            log_interval=0,
        )


# =============================================================================
# Tuning result
# =============================================================================


@dataclass
class TuneResult:
    """Output of BaseTuner.tune()."""

    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    all_trials: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Fit result
# =============================================================================


@dataclass
class FitResult:
    """Output of BaseTrainer.fit()."""

    model: Any
    best_iteration: int
    params: Dict[str, Any]
    seed: int
    evals_result: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[pl.DataFrame] = None
    checkpoint_path: Optional[Path] = None


# =============================================================================
# Evaluation result
# =============================================================================


@dataclass
class EvalResult:
    """Per-split evaluation output."""

    metrics: Dict[str, float]
    series: Dict[str, pl.Series] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    split: str = ""  # "train" / "val" / "test"


# =============================================================================
# State delta (for RollingState updates)
# =============================================================================


@dataclass
class StateDelta:
    """Incremental update produced by one split run, consumed by RollingState."""

    importance: Optional[np.ndarray] = None
    feature_hash: str = ""
    best_params: Optional[Dict[str, Any]] = None
    best_objective: Optional[float] = None


# =============================================================================
# Run output
# =============================================================================


@dataclass
class RunOutput:
    """Complete output of ModelPipeline.run() or EnsemblePipeline.run()."""

    best_params: Dict[str, Any]
    metrics: Dict[str, EvalResult]  # {"train": ..., "val": ..., "test": ...}
    importance: np.ndarray
    feature_hash: str
    tune_result: Optional[TuneResult] = None
    fit_result: Optional[FitResult] = None
    state_delta: Optional[StateDelta] = None
    artifacts: Optional[Dict[str, Path]] = None

    def get_state_delta(self) -> StateDelta:
        if self.state_delta is not None:
            return self.state_delta
        return StateDelta(
            importance=self.importance,
            feature_hash=self.feature_hash,
            best_params=self.best_params,
            best_objective=self.tune_result.best_value if self.tune_result else None,
        )
