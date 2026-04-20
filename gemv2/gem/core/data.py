"""
Core data container types.

All types here depend ONLY on numpy and polars -- no gem internal imports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# =============================================================================
# Split specification
# =============================================================================


@dataclass
class SplitSpec:
    """Defines which dates belong to train / val / test for one split."""

    split_id: int
    train_date_list: List[int]
    val_date_list: List[int]
    test_date_list: List[int]

    def get_all_dates_range(self) -> Tuple[int, int]:
        all_dates = self.train_date_list + self.val_date_list + self.test_date_list
        if not all_dates:
            raise ValueError(f"Split {self.split_id} has no dates in train/val/test.")
        return min(all_dates), max(all_dates)


@dataclass
class SplitPlan:
    """Output of SplitGenerator.generate() -- the plan for how to split data."""

    splitspec_list: List[SplitSpec]
    date_start: int
    date_end: int


# =============================================================================
# Data source specification
# =============================================================================


@dataclass
class SourceSpec:
    """Describes one data source (feather file, table layout, column mapping)."""

    name: str
    format: str = "feather"
    path: str = ""
    layout: str = "tabular"
    date_col: str = "date"
    code_col: str = "code"
    pivot: Optional[str] = None
    index_col: Optional[str] = None
    value_col: Optional[str] = None
    rename_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    """Specifies which sources map to X, y, and extra."""

    X_source_list: List[str] = field(default_factory=list)
    y_source_list: List[str] = field(default_factory=list)
    extra_source_list: List[str] = field(default_factory=list)
    key_cols: List[str] = field(default_factory=list)
    group_col: Optional[List[str]] = None


# =============================================================================
# Global dataset
# =============================================================================


@dataclass
class GlobalDataset:
    """
    Aligned global dataset.  Every SplitView is sliced from here via take().

    Attributes:
        keys:  (n_samples,) DataFrame with at least ``date_col`` and ``code_col``.
        X_full:  (n_samples, n_features) feature matrix.
        y_full:  (n_samples, n_labels) label matrix.
        feature_name_list:  length == n_features.
        label_name_list:  length == n_labels.
        extra:  optional auxiliary DataFrame (ret, liquidity, benchmark scores, ...).
    """

    keys: pl.DataFrame
    X_full: np.ndarray
    y_full: np.ndarray
    feature_name_list: List[str]
    label_name_list: List[str]
    date_col: str = "date"
    code_col: str = "code"
    extra: Optional[pl.DataFrame] = None

    _date_to_indices: Dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    # -- lifecycle --------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()
        self._build_date_index()

    def _validate(self) -> None:
        n_samples = self.keys.height

        if self.X_full.shape[0] != n_samples:
            raise ValueError(
                f"X_full row count ({self.X_full.shape[0]}) != keys ({n_samples})."
            )
        if self.y_full.shape[0] != n_samples:
            raise ValueError(
                f"y_full row count ({self.y_full.shape[0]}) != keys ({n_samples})."
            )
        if self.X_full.ndim != 2:
            raise ValueError(f"X_full must be 2D, got shape={self.X_full.shape}.")

        if self.y_full.ndim == 1:
            self.y_full = self.y_full.reshape(-1, 1)
        elif self.y_full.ndim != 2:
            raise ValueError(f"y_full must be 1D or 2D, got shape={self.y_full.shape}.")

        if self.X_full.shape[1] != len(self.feature_name_list):
            raise ValueError(
                f"X_full cols ({self.X_full.shape[1]}) != "
                f"feature_name_list ({len(self.feature_name_list)})."
            )
        if self.y_full.shape[1] != len(self.label_name_list):
            raise ValueError(
                f"y_full cols ({self.y_full.shape[1]}) != "
                f"label_name_list ({len(self.label_name_list)})."
            )

        if self.date_col not in self.keys.columns:
            raise ValueError(f"Missing required key column '{self.date_col}'.")
        if self.code_col not in self.keys.columns:
            raise ValueError(f"Missing required key column '{self.code_col}'.")

    def _build_date_index(self) -> None:
        self._date_to_indices.clear()
        date_arr = self.keys[self.date_col].to_numpy()
        for date in np.unique(date_arr):
            self._date_to_indices[int(date)] = np.flatnonzero(date_arr == date)

    # -- public API -------------------------------------------------------

    def get_indices_by_dates(self, dates: List[int]) -> np.ndarray:
        parts: List[np.ndarray] = []
        for date in dates:
            day_indices = self._date_to_indices.get(int(date))
            if day_indices is not None:
                parts.append(day_indices)
        if not parts:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts)

    def take(self, indices: np.ndarray) -> "SplitView":
        idx = np.asarray(indices, dtype=np.int64)
        idx_list = idx.tolist()
        return SplitView(
            indices=idx,
            X=self.X_full[idx],
            y=self.y_full[idx],
            keys=self.keys[idx_list],
            feature_names=self.feature_name_list.copy(),
            label_names=self.label_name_list.copy(),
            extra=self.extra[idx_list] if self.extra is not None else None,
        )

    # -- properties -------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.keys.height

    @property
    def n_features(self) -> int:
        return len(self.feature_name_list)

    @property
    def dates(self) -> np.ndarray:
        return self.keys[self.date_col].unique().sort().to_numpy()


# =============================================================================
# Split views
# =============================================================================


@dataclass
class SplitView:
    """A slice of GlobalDataset -- one of train / val / test."""

    indices: np.ndarray
    X: np.ndarray
    y: np.ndarray
    keys: pl.DataFrame
    feature_names: List[str]
    label_names: List[str]
    extra: Optional[pl.DataFrame] = None
    group: Optional[pl.DataFrame] = None

    @property
    def n_samples(self) -> int:
        return len(self.indices)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1


@dataclass
class SplitBundle:
    """Train + val + test triple BEFORE transforms."""

    train: SplitView
    val: SplitView
    test: SplitView
    split_spec: SplitSpec


@dataclass
class ProcessedBundle:
    """Train + val + test triple AFTER transforms.  X / y / feature_names may differ."""

    train: SplitView
    val: SplitView
    test: SplitView
    split_spec: SplitSpec
    feature_mask: Optional[np.ndarray] = None

    def get(self, split: str) -> SplitView:
        try:
            return {"train": self.train, "val": self.val, "test": self.test}[split]
        except KeyError as exc:
            raise ValueError(
                f"Unknown split '{split}', expected one of train/val/test."
            ) from exc
