from __future__ import annotations

import numpy as np
import polars as pl

EPS = 1e-6
K_MAD = 1.4826
CLIP = 5.0


def split_style_alpha_cols(fac_long: pl.DataFrame, date_range: tuple, n_style: int = 40) -> tuple[list[str], list[str]]:
    start, end = date_range
    sub = fac_long.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    all_dates = sorted(sub["date"].unique().to_list())
    sample_dates = [all_dates[i] for i in np.linspace(0, len(all_dates) - 1, min(30, len(all_dates))).astype(int)]

    fac_cols = [c for c in fac_long.columns if c not in ("date", "Code")]
    ranked = sub.filter(pl.col("date").is_in(sample_dates)).sort(["date", "Code"]).with_columns(
        [pl.col(c).rank().over("date").alias(c) for c in fac_cols]
    )

    stability = {c: [] for c in fac_cols}
    for i in range(len(sample_dates) - 1):
        a = ranked.filter(pl.col("date") == sample_dates[i]).sort("Code")
        b = ranked.filter(pl.col("date") == sample_dates[i + 1]).sort("Code")
        common = a.join(b, on="Code", suffix="_b")
        for c in fac_cols:
            x = common[c].to_numpy()
            y = common[f"{c}_b"].to_numpy()
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() > 10:
                stability[c].append(float(np.corrcoef(x[m], y[m])[0, 1]))

    scored = sorted([(c, float(np.mean(v)) if v else 0.0) for c, v in stability.items()], key=lambda t: t[1], reverse=True)
    style = [c for c, _ in scored[:n_style]]
    alpha = [c for c, _ in scored[n_style:]]
    return style, alpha


def mad_standardize_cross_section(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    if not cols:
        return df
    med = df.select([pl.col(c).median().alias(c) for c in cols]).row(0)
    filled = df.with_columns([pl.col(c).fill_null(med[i]).alias(c) for i, c in enumerate(cols)])
    arrs = filled.select(cols).to_numpy().astype(np.float32)
    m = np.asarray(med, dtype=np.float32)
    mad = np.median(np.abs(arrs - m), axis=0)
    std = arrs.std(axis=0, ddof=1)
    scale = np.where(K_MAD * mad < EPS, std + EPS, K_MAD * mad)
    z = np.clip((arrs - m) / scale, -CLIP, CLIP)
    return filled.with_columns([pl.Series(c, z[:, i], dtype=pl.Float32) for i, c in enumerate(cols)])


def industry_id_from_code(code: str) -> int:
    if code.startswith("000") or code.startswith("001"):
        return 0
    if code.startswith("002"):
        return 1
    if code.startswith("300"):
        return 2
    if code.startswith(("600", "601", "603", "605")):
        return 3
    if code.startswith("688"):
        return 4
    return 5


def precompute_ret_hist(label_long: pl.DataFrame, hist_len: int) -> dict:
    wide = label_long.pivot(values="label", index="date", on="Code").sort("date")
    dates = wide["date"].to_list()
    code_cols = [c for c in wide.columns if c != "date"]
    arr = wide.select(code_cols).to_numpy().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    T, N = arr.shape
    out = {}
    for t in range(T):
        lo = max(0, t - hist_len + 1)
        window = arr[lo:t + 1]
        if window.shape[0] < hist_len:
            window = np.vstack([np.zeros((hist_len - window.shape[0], N), dtype=np.float32), window])
        out[dates[t]] = (code_cols, window.T)
    return out
