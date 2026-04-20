"""
Label transformation module
Supports three label modes:
  - raw            : Original labels (unchanged)
  - residual       : Subtract daily cross-sectional mean (remove market-wide movements)
  - risk_adjusted  : Divide by individual stock rolling 60-day volatility (select stable alpha)
"""

import polars as pl


def transform_label(label_df: pl.DataFrame, mode: str) -> pl.DataFrame:
    """
    Transform labels.

    Parameters
    ----------
    label_df : Wide table, 'index' column is date, other columns are stock codes, values are original returns (pl.DataFrame)
    mode     : 'raw' | 'residual' | 'risk_adjusted'

    Returns
    -------
    Transformed wide table, same format as input (including 'index' column) (pl.DataFrame)
    """
    if mode == "raw":
        return label_df.clone()

    # Ensure date column is Datetime type and sort
    df = label_df.clone()
    df = df.sort("index")
    stock_cols = [c for c in df.columns if c != "index"]

    if mode == "residual":
        # Subtract daily cross-sectional mean, remove market-wide movement impact
        result = df.with_columns(
            pl.mean_horizontal(*[pl.col(c) for c in stock_cols]).alias("_row_mean")
        ).with_columns(
            *[(pl.col(c) - pl.col("_row_mean")).alias(c) for c in stock_cols]
        ).drop("_row_mean")

    elif mode == "risk_adjusted":
        # Divide by individual stock rolling 60-day volatility (5-day lag to avoid future information leakage)
        # polars multi-threaded rolling_std: parallel computation for all stock columns
        lagged_vol = df.select([
            pl.col("index"),
            *[
                pl.col(c).rolling_std(window_size=60, min_samples=20).shift(5).alias(c)
                for c in stock_cols
            ],
        ])
        # Divide by (vol + eps), then clip
        result = df.select([
            pl.col("index"),
            *[
                (pl.col(c) / (lagged_vol[c] + 1e-6)).clip(-10, 10).alias(c)
                for c in stock_cols
            ],
        ])

    elif mode == "winsorized":
        # Cross-sectional 3-sigma winsorization per date row, then return
        # Clip each row's values at mean ± 3*std before downstream z-score normalization
        row_mean = pl.mean_horizontal(*[pl.col(c) for c in stock_cols])
        row_std = pl.concat_list([pl.col(c) for c in stock_cols]).list.eval(
            pl.element().std(ddof=1)
        ).list.first()
        # Compute bounds
        result = df.with_columns([
            row_mean.alias("_row_mean"),
            row_std.alias("_row_std"),
        ]).with_columns([
            (pl.col("_row_mean") - 3.0 * pl.col("_row_std")).alias("_lo"),
            (pl.col("_row_mean") + 3.0 * pl.col("_row_std")).alias("_hi"),
        ]).with_columns([
            pl.col(c).clip(pl.col("_lo"), pl.col("_hi")).alias(c)
            for c in stock_cols
        ]).drop(["_row_mean", "_row_std", "_lo", "_hi"])

    elif mode == "rank_transform":
        # Cross-sectional rank transformation: y = rank(return) / N per date row
        # Maps to uniform [0, 1] distribution, fully robust to outliers
        import numpy as np
        dates_idx = df["index"]
        data = df.select(stock_cols).to_numpy()
        ranked = np.empty_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            row = data[i]
            valid_mask = ~np.isnan(row)
            n_valid = valid_mask.sum()
            if n_valid > 0:
                from scipy.stats import rankdata
                ranked[i, valid_mask] = rankdata(row[valid_mask]) / n_valid
                ranked[i, ~valid_mask] = np.nan
            else:
                ranked[i, :] = np.nan
        result = pl.DataFrame(
            {"index": dates_idx, **{c: ranked[:, j] for j, c in enumerate(stock_cols)}}
        )

    else:
        raise ValueError(
            f"Unknown label mode: {mode}, supports "
            f"'raw' / 'residual' / 'risk_adjusted' / 'winsorized' / 'rank_transform'"
        )

    return result
