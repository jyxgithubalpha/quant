"""
Factor IC momentum selection module
For each quarter, selects factors based on factor RankIC momentum from the previous lookback_months,
keeping factors with |short_ic| > ic_threshold and momentum scores ranked in top keep_ratio.

Key design: Use matrix operations (rank → corr) to complete cross-sectional Spearman for all factors at once,
avoiding individual for-loop calculations for 941 factors.
"""

import numpy as np
import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import rankdata

from model_core import _pl_melt_join


def _fast_spearman_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of Spearman correlation between each column of X and y.

    Parameters
    ----------
    X : (n_stocks, n_factors) ndarray
    y : (n_stocks,) ndarray

    Returns
    -------
    corrs : (n_factors,) ndarray
    """
    n = len(y)
    if n < 2:
        return np.zeros(X.shape[1])

    # Rank y
    y_rank = rankdata(y, method="average")
    y_rank -= y_rank.mean()
    y_std = y_rank.std()
    if y_std < 1e-9:
        return np.zeros(X.shape[1])
    y_rank /= y_std

    # Rank each column of X (vectorized)
    X_rank = np.apply_along_axis(lambda col: rankdata(col, method="average"), 0, X)
    X_rank -= X_rank.mean(axis=0, keepdims=True)
    X_std = X_rank.std(axis=0)
    # Avoid division by zero
    safe_std = np.where(X_std < 1e-9, 1.0, X_std)
    X_rank = X_rank / safe_std

    corrs = (X_rank * y_rank[:, None]).mean(axis=0)
    corrs = np.where(X_std < 1e-9, 0.0, corrs)
    return corrs


def _compute_monthly_factor_ic(fac_df: pl.DataFrame, label_df: pl.DataFrame,
                                feature_cols: list) -> pl.DataFrame:
    """
    Calculate cross-sectional RankIC for each month and each factor.

    Returns
    -------
    monthly_ic : pl.DataFrame, containing 'year_month' column + feature_cols columns, values=RankIC
    """
    merged = (
        _pl_melt_join(fac_df, label_df)
        .filter(pl.col("label").is_not_null())
        .sort("date")
    )

    # Add year_month column for grouping
    merged = merged.with_columns(
        (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("year_month")
    )

    # Ensure feature_cols are all in merged
    available = [c for c in feature_cols if c in merged.columns]

    records = []
    for month_grp in merged.partition_by("year_month", maintain_order=True):
        ym = month_grp["year_month"][0]
        month_grp = month_grp.drop_nulls(subset=available)
        if len(month_grp) < 5:
            continue

        # Calculate cross-sectional RankIC for each day, then take mean
        day_ics = []
        for day_grp in month_grp.partition_by("date", maintain_order=True):
            if len(day_grp) < 5:
                continue
            X_d = day_grp.select(available).to_numpy().astype(np.float64)
            y_d = day_grp["label"].to_numpy().astype(np.float64)
            day_ics.append(_fast_spearman_matrix(X_d, y_d))

        if not day_ics:
            continue
        month_ic = np.mean(day_ics, axis=0)  # (n_factors,)
        records.append({"year_month": ym, **dict(zip(available, month_ic))})

    if not records:
        return pl.DataFrame(schema={"year_month": pl.Int64, **{c: pl.Float64 for c in available}})

    monthly_ic = pl.DataFrame(records).sort("year_month")
    return monthly_ic


def select_features_by_timing(fac_df: pl.DataFrame, label_df: pl.DataFrame,
                               feature_cols: list,
                               test_year: int, test_quarter: int,
                               lookback_months: int = 6,
                               keep_ratio: float = 0.7,
                               ic_threshold: float = 0.005) -> list:
    """
    Select factor subset from feature_cols based on factor IC momentum.

    Logic:
    1. Take historical data from lookback_months before test quarter starts
    2. Calculate cross-sectional RankIC for all factors by month
    3. short_ic  = mean of recent 3 months
       long_ic   = mean of recent lookback_months
       momentum  = short_ic + 0.2 * (short_ic > long_ic bonus)
    4. Filter factors with |short_ic| <= ic_threshold
    5. Sort by momentum descending, keep top keep_ratio proportion

    Parameters
    ----------
    fac_df, label_df : pl.DataFrame Original factor/label data (full time range)
    feature_cols     : Initial factor list
    test_year, test_quarter : Current test quarter
    lookback_months  : How many months to look back (default 6)
    keep_ratio       : Keep ratio (default 0.7)
    ic_threshold     : IC absolute value threshold (default 0.005)

    Returns
    -------
    selected_features : list, filtered factor names
    """
    # Determine historical range (lookback_months before test quarter starts)
    test_start_month = (test_quarter - 1) * 3 + 1
    test_start_dt = datetime(test_year, test_start_month, 1)
    history_end_dt = test_start_dt - relativedelta(months=1)  # One month before test quarter
    history_start_dt = test_start_dt - relativedelta(months=lookback_months)

    # Month-end date
    hist_end_date = (history_end_dt + relativedelta(months=1)) - relativedelta(days=1)
    hist_start_date = history_start_dt

    # Filter historical data
    fac = fac_df  # Already datetime from _read_feather
    hist_fac = fac.filter(
        (pl.col("date") >= hist_start_date) & (pl.col("date") <= hist_end_date)
    )

    label = label_df  # Already datetime from _read_feather
    hist_label = label.filter(
        (pl.col("index") >= hist_start_date) & (pl.col("index") <= hist_end_date)
    )

    if hist_fac.is_empty() or hist_label.is_empty():
        print(f"[factor_timing] Insufficient historical data, returning all {len(feature_cols)} factors")
        return list(feature_cols)

    # Calculate monthly factor IC
    monthly_ic = _compute_monthly_factor_ic(hist_fac, hist_label, feature_cols)

    fac_cols = [c for c in monthly_ic.columns if c != "year_month"]
    if monthly_ic.is_empty() or not fac_cols:
        print(f"[factor_timing] Cannot calculate IC, returning all factors")
        return list(feature_cols)

    # Calculate IC momentum (use numpy for vectorized computation)
    ic_matrix = monthly_ic.select(fac_cols).to_numpy()  # (n_months, n_factors)
    short_ic = np.nanmean(ic_matrix[-3:], axis=0) if len(ic_matrix) >= 3 else np.nanmean(ic_matrix, axis=0)
    long_ic = np.nanmean(ic_matrix, axis=0)

    improving_bonus = (np.abs(short_ic) > np.abs(long_ic)).astype(float) * 0.2
    momentum = np.abs(short_ic) + improving_bonus

    # Filter factors with insignificant IC
    sig_mask = np.abs(short_ic) > ic_threshold
    if not sig_mask.any():
        print(f"[factor_timing] All factor IC below threshold {ic_threshold}, returning all factors")
        return list(feature_cols)

    # Sort by momentum descending, keep top keep_ratio
    sig_indices = np.where(sig_mask)[0]
    sig_momentum = momentum[sig_indices]
    sig_names = [fac_cols[i] for i in sig_indices]

    n_keep = max(1, int(len(sig_momentum) * keep_ratio))
    top_idx = np.argsort(-sig_momentum)[:n_keep]
    selected = [sig_names[i] for i in top_idx]

    print(f"[factor_timing] {test_year}-Q{test_quarter}: "
          f"{len(feature_cols)} → {len(selected)} factors "
          f"(|short_ic|>{ic_threshold}, top {keep_ratio*100:.0f}%)")
    return selected
