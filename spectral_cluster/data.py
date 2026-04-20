"""
data.py — Data loading, merging, cross-sectional normalization, splitting,
           and benchmark ensemble construction.
"""
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime


# ========================= Core data preparation =========================

def prepare_data(fac_df: pl.DataFrame, label_df: pl.DataFrame,
                 liquid_df: pl.DataFrame, clip_range: float = 5.0):
    """
    Merge factor / label / liquidity data and apply cross-sectional
    normalization (median-MAD for factors, winsorize + zscore for labels).
    Returns (polars DataFrame, list of factor column names).
    """
    fac_long_df = fac_df
    label_long_df = (
        label_df
        .unpivot(index=["index"], variable_name="Code", value_name="label")
        .rename({"index": "date"})
    )
    liquid_long_df = (
        liquid_df
        .unpivot(index=["index"], variable_name="Code", value_name="liquid")
        .rename({"index": "date"})
    )

    merged_df = fac_long_df.join(label_long_df, on=["date", "Code"], how="inner")
    merged_df = merged_df.join(liquid_long_df, on=["date", "Code"], how="inner")
    merged_df = merged_df.drop_nulls().sort(["date", "Code"])

    # Unify date column to pl.Date
    if merged_df["date"].dtype in (pl.Utf8, pl.String):
        merged_df = merged_df.with_columns(pl.col("date").str.to_date("%Y%m%d"))
    elif merged_df["date"].dtype != pl.Date:
        merged_df = merged_df.with_columns(pl.col("date").cast(pl.Date))

    fac_cols = [c for c in merged_df.columns
                if c not in ["date", "Code", "label", "liquid"]]

    # Filter dates with fewer than 5 samples
    merged_df = (
        merged_df
        .with_columns(pl.len().over("date").alias("__cnt__"))
        .filter(pl.col("__cnt__") >= 5)
        .drop("__cnt__")
    )

    # --- Cross-sectional normalization: factors (median-MAD) ---
    # Step 1: fill nulls with cross-sectional median, then 0
    merged_df = merged_df.with_columns([
        pl.col(c).fill_null(pl.col(c).median().over("date")).fill_null(0.0).alias(c)
        for c in fac_cols
    ])

    # Step 2: median_norm — (x - median) / scale, scale = 1.4826*MAD or fallback std
    EPS = 1e-6
    med_col_map = {c: f"__{c}_med" for c in fac_cols}
    abs_dev_col_map = {c: f"__{c}_abs_dev" for c in fac_cols}

    merged_df = merged_df.with_columns([
        pl.col(c).median().over("date").alias(med_col_map[c])
        for c in fac_cols
    ])
    merged_df = merged_df.with_columns([
        (pl.col(c) - pl.col(med_col_map[c])).abs().alias(abs_dev_col_map[c])
        for c in fac_cols
    ])

    norm_exprs = []
    for c in fac_cols:
        med = pl.col(med_col_map[c])
        mad = pl.col(abs_dev_col_map[c]).median().over("date")
        scale_mad = mad * 1.4826
        std_fb = pl.col(c).std().over("date") + EPS
        scale = (
            pl.when(scale_mad > EPS).then(scale_mad)
            .otherwise(pl.when(std_fb > EPS).then(std_fb).otherwise(1.0))
        )
        normed = (
            ((pl.col(c) - med) / scale)
            .clip(-clip_range, clip_range)
            .fill_nan(0.0)
            .fill_null(0.0)
        )
        norm_exprs.append(normed.alias(c))
    merged_df = merged_df.with_columns(norm_exprs)
    merged_df = merged_df.drop(
        list(med_col_map.values()) + list(abs_dev_col_map.values())
    )

    # --- Cross-sectional normalization: label (winsorize + zscore) ---
    q01 = pl.col("label").quantile(0.01).over("date")
    q99 = pl.col("label").quantile(0.99).over("date")
    merged_df = merged_df.with_columns(
        pl.when(pl.col("label") < q01).then(q01)
        .when(pl.col("label") > q99).then(q99)
        .otherwise(pl.col("label"))
        .alias("label")
    )
    lmean = pl.col("label").mean().over("date")
    lstd = pl.col("label").std().over("date")
    merged_df = merged_df.with_columns(
        pl.when(lstd > EPS)
        .then((pl.col("label") - lmean) / lstd)
        .otherwise(0.0)
        .alias("label")
    )

    print(f"Data ready: {merged_df.shape[0]} rows, {len(fac_cols)} factors")
    return merged_df, fac_cols


def split_by_date(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """Filter by date range (YYYYMMDD strings, date column is pl.Date)."""
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    return df.filter((pl.col("date") >= s) & (pl.col("date") <= e))


def subsample(df: pl.DataFrame, n: int, seed: int = 42) -> pl.DataFrame:
    """Randomly sample at most n rows."""
    if len(df) <= n:
        return df
    return df.sample(n=n, seed=seed)


# ========================= Benchmark helpers =========================

def load_bench_wide(path: str) -> pd.DataFrame:
    """Load a feather wide table -> pandas DataFrame (index=date_str, cols=stock)."""
    df = pd.read_feather(path)
    idx_col = "date" if "date" in df.columns else "index"
    df = df.set_index(idx_col)
    df.index = df.index.astype(str)
    return df


def zscore_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score normalization for a wide DataFrame."""
    return df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)


def build_bench_ensemble(bench_wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Z-score each bench wide table cross-sectionally, then equal-weight average.
    Returns a wide DataFrame (index=date_str, cols=stock).
    """
    normed = [zscore_wide(df) for df in bench_wide.values()]

    # Align index & columns
    common_idx = normed[0].index
    common_cols = normed[0].columns
    for n in normed[1:]:
        common_idx = common_idx.intersection(n.index)
        common_cols = common_cols.intersection(n.columns)

    aligned = [n.loc[common_idx, common_cols] for n in normed]
    return sum(aligned) / len(aligned)


def build_ensemble_with_model(bench_wide: dict[str, pd.DataFrame],
                              model_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight z-score ensemble of bench1-6 + a model score table.
    All 7 wide tables are z-scored cross-sectionally then averaged.
    """
    all_normed = [zscore_wide(df) for df in bench_wide.values()]
    all_normed.append(zscore_wide(model_wide))

    common_idx = all_normed[0].index
    common_cols = all_normed[0].columns
    for n in all_normed[1:]:
        common_idx = common_idx.intersection(n.index)
        common_cols = common_cols.intersection(n.columns)

    aligned = [n.loc[common_idx, common_cols] for n in all_normed]
    return sum(aligned) / len(aligned)
