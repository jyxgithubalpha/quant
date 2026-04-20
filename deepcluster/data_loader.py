"""Data loading and preprocessing for DeepCluster V2.

Follows tangles_ens pattern: Polars-based, feather I/O, long format.
Factor standardization reuses xgboost MAD approach.
"""

from datetime import datetime

import numpy as np
import polars as pl

from config import FAC_PATH, LABEL_PATH, LIQUID_PATH, QUARTERS, SPLIT_CONFIG


# ── Quarter date utilities ──────────────────────────────────────────────

QUARTER_STARTS = {1: "0101", 2: "0401", 3: "0701", 4: "1001"}
QUARTER_ENDS = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}


def quarter_to_dt(year: int, q: int, start: bool = True) -> datetime:
    s = f"{year}{QUARTER_STARTS[q] if start else QUARTER_ENDS[q]}"
    return datetime.strptime(s, "%Y%m%d")


def get_split_boundaries(quarter_idx: int) -> dict:
    """Compute train/val/test date boundaries for a given test quarter index."""
    test_year, test_q = QUARTERS[quarter_idx]
    test_start = quarter_to_dt(test_year, test_q, start=True)
    test_end = quarter_to_dt(test_year, test_q, start=False)

    gap = SPLIT_CONFIG["gap_days"]
    val_n = SPLIT_CONFIG["valid_quarters"]

    # Validation: val_n quarters before test quarter
    val_start_idx = max(0, quarter_idx - val_n)
    vy, vq = QUARTERS[val_start_idx]
    val_start = quarter_to_dt(vy, vq, start=True)

    from datetime import timedelta
    val_end = test_start - timedelta(days=gap)
    train_end = val_start - timedelta(days=gap)

    return {
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


# ── Data loading ────────────────────────────────────────────────────────

def _ensure_datetime(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if df[col].dtype == pl.String:
        return df.with_columns(pl.col(col).str.strptime(pl.Datetime("us"), "%Y%m%d"))
    if df[col].dtype != pl.Datetime("us"):
        return df.with_columns(pl.col(col).cast(pl.Datetime("us")))
    return df


def load_factors() -> pl.DataFrame:
    """Load factor data (long: date, Code, fac1, fac2, ...).

    Returns sorted by [date, Code].
    """
    df = pl.read_ipc(FAC_PATH, memory_map=False)
    df = _ensure_datetime(df, "date")
    if "Code" in df.columns and df["Code"].dtype == pl.String:
        df = df.with_columns(pl.col("Code").cast(pl.Categorical))
    return df.sort(["date", "Code"])


def load_labels() -> pl.DataFrame:
    """Load label file (wide → long): date, Code, label."""
    df = pl.read_ipc(LABEL_PATH, memory_map=False)
    label_cols = [c for c in df.columns if c != "index"]
    long = df.unpivot(
        on=label_cols, index="index",
        variable_name="Code", value_name="label",
    ).rename({"index": "date"})
    long = _ensure_datetime(long, "date")
    return long.drop_nulls(subset=["label"])


def load_liquidity() -> pl.DataFrame:
    """Load liquidity file (wide → long): date, Code, can_trade."""
    df = pl.read_ipc(LIQUID_PATH, memory_map=False)
    liq_cols = [c for c in df.columns if c != "index"]
    long = df.unpivot(
        on=liq_cols, index="index",
        variable_name="Code", value_name="can_trade",
    ).rename({"index": "date"})
    long = _ensure_datetime(long, "date")
    return long


def load_liquidity_wide():
    """Load liquidity as pandas wide DataFrame (for backtest eval)."""
    import pandas as pd
    df = pd.read_feather(LIQUID_PATH).set_index("index")
    df.index = df.index.astype(str)
    return df


def load_labels_wide():
    """Load labels as pandas wide DataFrame (for backtest eval)."""
    import pandas as pd
    df = pd.read_feather(LABEL_PATH).set_index("index")
    df.index = df.index.astype(str)
    return df


# ── MAD standardization ────────────────────────────────────────────────

def mad_standardize(df: pl.DataFrame, feat_cols: list[str]) -> pl.DataFrame:
    """Cross-sectional MAD standardization (same as xgboost/model_core.py).

    For each date cross-section:
        x_norm = (x - median) / (1.4826 * MAD + eps)
    Falls back to std if MAD ≈ 0.
    """
    EPS = 1e-6
    K = 1.4826
    ldf = df.lazy()

    # Filter dates with < 5 stocks
    valid_dates = (
        ldf.group_by("date").len()
        .filter(pl.col("len") >= 5)
        .select("date")
    )
    ldf = ldf.join(valid_dates, on="date", how="semi")

    # Per-date median
    med = ldf.group_by("date").agg([
        pl.col(c).median().alias(f"__{c}_med") for c in feat_cols
    ])
    ldf = ldf.join(med, on="date", how="left")

    # Fill nulls with median
    ldf = ldf.with_columns([
        pl.col(c).fill_null(pl.col(f"__{c}_med")).alias(f"__{c}_f")
        for c in feat_cols
    ])

    # Per-date MAD and std
    mad_std = ldf.group_by("date").agg(
        [(pl.col(f"__{c}_f") - pl.col(f"__{c}_med")).abs().median().alias(f"__{c}_mad")
         for c in feat_cols]
        + [pl.col(f"__{c}_f").std(ddof=1).alias(f"__{c}_std")
           for c in feat_cols]
    )
    ldf = ldf.join(mad_std, on="date", how="left")

    # Normalize
    ldf = ldf.with_columns([
        ((pl.col(f"__{c}_f") - pl.col(f"__{c}_med"))
         / pl.when(K * pl.col(f"__{c}_mad") < EPS)
             .then(pl.col(f"__{c}_std") + EPS)
             .otherwise(K * pl.col(f"__{c}_mad"))
         ).alias(c)
        for c in feat_cols
    ])

    # Drop temp columns
    tmp = []
    for c in feat_cols:
        tmp.extend([f"__{c}_med", f"__{c}_f", f"__{c}_mad", f"__{c}_std"])
    return ldf.drop(tmp).collect()


# ── Split preparation ──────────────────────────────────────────────────

def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Get factor column names (everything except date, Code, label)."""
    exclude = {"date", "Code", "label"}
    return [c for c in df.columns if c not in exclude]


def prepare_quarter_data(
    fac_df: pl.DataFrame,
    label_df: pl.DataFrame,
    quarter_idx: int,
) -> dict:
    """Prepare train/val/test data for one quarter.

    Returns dict with:
        X_train, y_train, dates_train, codes_train,
        X_val, y_val, dates_val, codes_val,
        X_test, y_test, dates_test, codes_test,
        feat_cols
    """
    bounds = get_split_boundaries(quarter_idx)
    feat_cols = get_feature_cols(fac_df)

    # Merge factors + labels
    merged = fac_df.join(
        label_df.select(["date", "Code", "label"]),
        on=["date", "Code"], how="inner",
    ).filter(pl.col("label").is_not_null())

    # MAD standardize on full data (each cross-section independently)
    merged = mad_standardize(merged, feat_cols)

    def _extract(df: pl.DataFrame, start, end):
        subset = df.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        ).sort(["date", "Code"])
        if subset.is_empty():
            n_feat = len(feat_cols)
            return (np.empty((0, n_feat), dtype=np.float32),
                    np.empty(0, dtype=np.float32), [], [])
        X = subset.select(feat_cols).to_numpy().astype(np.float32)
        y = subset["label"].to_numpy().astype(np.float32)
        dates = subset["date"].to_list()
        codes = subset["Code"].cast(pl.String).to_list()
        return X, y, dates, codes

    # Train: all data up to train_end
    first_date = merged["date"].min()
    X_tr, y_tr, d_tr, c_tr = _extract(merged, first_date, bounds["train_end"])
    X_va, y_va, d_va, c_va = _extract(merged, bounds["val_start"], bounds["val_end"])
    X_te, y_te, d_te, c_te = _extract(merged, bounds["test_start"], bounds["test_end"])

    print(f"  Quarter {QUARTERS[quarter_idx]}: "
          f"train={X_tr.shape[0]}, val={X_va.shape[0]}, test={X_te.shape[0]}")

    return {
        "X_train": X_tr, "y_train": y_tr, "dates_train": d_tr, "codes_train": c_tr,
        "X_val": X_va, "y_val": y_va, "dates_val": d_va, "codes_val": c_va,
        "X_test": X_te, "y_test": y_te, "dates_test": d_te, "codes_test": c_te,
        "feat_cols": feat_cols,
    }


if __name__ == "__main__":
    print("Loading factors...")
    fac = load_factors()
    print(f"Factors shape: {fac.shape}, columns: {len(get_feature_cols(fac))}")
    print(f"Date range: {fac['date'].min()} ~ {fac['date'].max()}")

    print("Loading labels...")
    lab = load_labels()
    print(f"Labels: {lab.shape}")

    print("Loading liquidity...")
    liq = load_liquidity()
    print(f"Liquidity: {liq.shape}")
