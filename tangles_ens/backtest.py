"""Backtest runner: evaluate tangle ensemble vs equal-weight baseline.

Evaluation approach aligned with bench notebook:
- Daily rebalance with liquidity constraint (top-K stocks, fixed capital)
- Quarterly metrics breakdown
- Diversity entropy & cross-model correlation
- Formatted comparison tables
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")

import numpy as np
import pandas as pd
import polars as pl
from numpy.linalg import eigh
from scipy.stats import spearmanr
from tqdm import tqdm

from config import (
    BENCH_PATHS, EVAL_START, EVAL_END, LOOKBACK_DAYS,
    PORTFOLIO_CAPITAL, MAX_STOCKS, LABEL_PATH, LIQUID_PATH,
)
from data_loader import load_data
from tangle_ensemble import ensemble_one_date

BENCH_COLS = sorted(BENCH_PATHS.keys())


# ── Evaluation utilities (bench notebook style) ──────────────────────────


def get_ret_ic(
    score: pd.DataFrame,
    ret_data: pd.DataFrame,
    liquid_data: pd.DataFrame,
    start: str = EVAL_START,
    end: str = EVAL_END,
    money: float = PORTFOLIO_CAPITAL,
    max_stocks: int = MAX_STOCKS,
    print_quarterly: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Daily rebalance portfolio return and IC with liquidity constraint.

    Args:
        score: wide (index=date, columns=stock_code, values=score).
        ret_data: wide daily returns (decimal, ×100 inside).
        liquid_data: wide tradeable amount per stock.

    Returns:
        (daily_return_pct, daily_ic) as pd.Series indexed by date.
    """
    label_ret = []
    ic = []
    datelist = []

    for date in score.loc[start:end].index:
        datelist.append(date)
        code_rank = score.loc[date].dropna().sort_values(ascending=False)
        ret = ret_data.loc[date].reindex(code_rank.index).fillna(0) * 100
        liquid = liquid_data.loc[date].reindex(code_rank.index).fillna(0)

        total_hold = 0.0
        total_earned = 0.0
        for num, code in enumerate(code_rank.index):
            if num >= max_stocks:
                break
            if (money - total_hold) < 1:
                break
            hold_money = min(money - total_hold, liquid[code])
            total_hold += hold_money
            total_earned += ret[code] * hold_money

        label_ret.append(total_earned / money)
        ic.append(code_rank.corr(ret))

    ic = pd.Series(ic, index=datelist, dtype="float")
    label_ret = pd.Series(label_ret, index=datelist, dtype="float")

    if print_quarterly:
        print_quarterly_metrics(label_ret, ic)
    return label_ret, ic


def print_quarterly_metrics(ret: pd.Series, ic: pd.Series):
    """Print quarterly average return and IC."""
    ret_dt = ret.copy()
    ic_dt = ic.copy()
    ret_dt.index = pd.to_datetime(ret_dt.index, format="%Y%m%d")
    ic_dt.index = pd.to_datetime(ic_dt.index, format="%Y%m%d")

    q_ret = ret_dt.groupby(pd.Grouper(freq="QE")).mean()
    q_ic = ic_dt.groupby(pd.Grouper(freq="QE")).mean()
    q_ret.index = [f"{idx.year}Q{idx.quarter}" for idx in q_ret.index]
    q_ic.index = [f"{idx.year}Q{idx.quarter}" for idx in q_ic.index]

    df = pd.DataFrame({
        "Quarterly Avg Return(%)": q_ret.round(4),
        "Quarterly Avg IC": q_ic.round(4),
    })
    print("\n===== Quarterly Metrics =====")
    print(df)
    print("=============================\n")


def get_metrics(
    score: pd.DataFrame,
    ret_data: pd.DataFrame,
    liquid_data: pd.DataFrame,
    start: str = EVAL_START,
    end: str = EVAL_END,
    money: float = PORTFOLIO_CAPITAL,
    max_stocks: int = MAX_STOCKS,
    print_quarterly: bool = True,
) -> dict:
    """Compute IC, ICIR, top_return, top_return_stability."""
    ret_list, ic_list = get_ret_ic(
        score, ret_data, liquid_data,
        start=start, end=end, money=money, max_stocks=max_stocks,
        print_quarterly=print_quarterly,
    )
    return {
        "IC": ic_list.mean(),
        "ICIR": ic_list.mean() / ic_list.std() if ic_list.std() != 0 else 0,
        "top_return": ret_list.mean(),
        "top_return_stability": ret_list.mean() / ret_list.std() if ret_list.std() != 0 else 0,
    }


def print_metrics(
    model_metrics: dict,
    bench_metrics: dict | None = None,
    title: str = "Model Evaluation",
):
    """Print formatted metric comparison table."""
    keys = list(model_metrics.keys())
    df = pd.DataFrame(index=keys)
    df.index.name = "Metric"
    df["Model"] = [model_metrics[k] for k in keys]
    if bench_metrics is not None:
        df["Bench"] = [bench_metrics.get(k) for k in keys]
        df["Improve(%)"] = ((df["Model"] - df["Bench"]) / df["Bench"].abs()) * 100
    print(f"\n{title}")
    print(df.round(4))


def diversity_entropy(score_row):
    """Diversity entropy via eigenvalue decomposition of covariance matrix."""
    vals = score_row.values if isinstance(score_row, pd.DataFrame) else score_row
    m = vals.shape[1]
    cov = np.cov(vals, rowvar=False)
    eigenvalues, _ = eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    normed = eigenvalues / (np.sum(eigenvalues) + 1e-10)
    entropy = -sum(p * np.log(p) for p in normed if p > 1e-10)
    return entropy / np.log(m)


def ensemble_scores(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize and stack score DataFrames for correlation analysis."""
    parts = []
    for i, df in enumerate(dfs, 1):
        normed = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        parts.append(normed.stack().rename(f"score{i}"))
    return pd.concat(parts, axis=1).dropna()


# ── Wide DataFrame helpers ───────────────────────────────────────────────


def load_wide_pandas(path: str, index_col: str = "index") -> pd.DataFrame:
    """Load feather to wide pandas DataFrame with string index."""
    df = pd.read_feather(path).set_index(index_col)
    df.index = df.index.astype(str)
    return df


def compute_tangle_scores_wide(
    df: pl.DataFrame,
    dates: list,
    eval_dates: list,
    lookback_days: int,
) -> pd.DataFrame:
    """Run tangle ensemble for each eval date → wide score DataFrame."""
    date_idx = {d: i for i, d in enumerate(dates)}
    records = []

    for d in tqdm(eval_dates, desc="Tangle Ensemble"):
        idx = date_idx[d]
        date_df = df.filter(pl.col("date") == d)
        if date_df.height < 100:
            continue

        lookback_start = max(0, idx - lookback_days)
        hist_df = df.filter(pl.col("date").is_in(dates[lookback_start:idx]))

        ens_df = ensemble_one_date(date_df, hist_df, BENCH_COLS)
        for code, score in zip(ens_df["code"].to_list(), ens_df["ensemble_score"].to_list()):
            records.append((d, code, score))

    pdf = pd.DataFrame(records, columns=["date", "code", "score"])
    wide = pdf.pivot(index="date", columns="code", values="score")
    return wide.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


# ── Main ─────────────────────────────────────────────────────────────────


def run_backtest():
    print("Loading data...")
    df = load_data()
    dates = sorted(df["date"].unique().to_list())
    eval_dates = [d for d in dates if EVAL_START <= d <= EVAL_END]
    print(f"Eval period: {eval_dates[0]} ~ {eval_dates[-1]}, {len(eval_dates)} dates")

    # Load wide label / liquidity (pandas)
    ret_data = load_wide_pandas(LABEL_PATH)
    liquid_data = load_wide_pandas(LIQUID_PATH)

    # Load individual bench scores (wide pandas)
    bench_wide = {}
    for name, path in BENCH_PATHS.items():
        bench_wide[name] = pd.read_feather(path).set_index("date")
        bench_wide[name].index = bench_wide[name].index.astype(str)

    # ── 1. Bench baseline ──
    print("\n" + "=" * 60)
    print("BENCH BASELINE (Equal Weight Ensemble)")
    print("=" * 60)

    bench_ens = ensemble_scores(*bench_wide.values())
    de = bench_ens.groupby("date").apply(diversity_entropy)
    print(f"Sub-model diversity entropy: {de.mean():.4f}")
    print("Sub-model correlation matrix:")
    print(bench_ens.groupby("date").corr().unstack().mean().unstack().round(4))

    bench_all = bench_ens.mean(axis=1).unstack()
    bench_metrics = get_metrics(bench_all, ret_data, liquid_data, print_quarterly=True)
    print_metrics(bench_metrics, title="Bench Equal Weight Evaluation")

    # ── 2. Tangle ensemble ──
    print("\n" + "=" * 60)
    print("TANGLE ENSEMBLE")
    print("=" * 60)

    tangle_wide = compute_tangle_scores_wide(df, dates, eval_dates, LOOKBACK_DAYS)
    tangle_metrics = get_metrics(tangle_wide, ret_data, liquid_data, print_quarterly=True)
    print_metrics(tangle_metrics, bench_metrics, title="Tangle Ensemble vs Bench_Equal")

    # Correlation with bench
    common_idx = tangle_wide.index.intersection(bench_all.index)
    common_cols = tangle_wide.columns.intersection(bench_all.columns)
    if len(common_idx) > 0 and len(common_cols) > 0:
        corr = tangle_wide.loc[common_idx, common_cols].corrwith(
            bench_all.loc[common_idx, common_cols], axis=1,
        )
        print(f"\nTangle Ensemble vs Bench_Equal correlation: {corr.mean():.4f}")

    # ── 3. Diversity with tangle added ──
    print("\n" + "=" * 60)
    print("ENSEMBLE WITH TANGLE ADDED")
    print("=" * 60)

    tangle_aligned = tangle_wide.reindex(
        index=bench_all.index, columns=bench_all.columns,
    )
    new_ens = ensemble_scores(*bench_wide.values(), tangle_aligned)
    de_new = new_ens.groupby("date").apply(diversity_entropy)
    print(f"Diversity entropy with Tangle: {de_new.mean():.4f} (was: {de.mean():.4f})")
    print("Correlation matrix with Tangle added:")
    print(new_ens.groupby("date").corr().unstack().mean().unstack().round(4))

    new_all = new_ens.mean(axis=1).unstack()
    new_metrics = get_metrics(new_all, ret_data, liquid_data, print_quarterly=True)
    print_metrics(new_metrics, bench_metrics, title="Ensemble+Tangle vs Bench_Equal")

    # ── 4. Individual bench ──
    print("\n" + "=" * 60)
    print("INDIVIDUAL BENCH PERFORMANCE")
    print("=" * 60)
    for name, score_df in bench_wide.items():
        m = get_metrics(score_df, ret_data, liquid_data, print_quarterly=False)
        print(f"  {name}: IC={m['IC']:.4f}  ICIR={m['ICIR']:.4f}  "
              f"top_return={m['top_return']:.4f}%  stability={m['top_return_stability']:.4f}")
    print("=" * 60)

    return {
        "tangle_wide": tangle_wide,
        "bench_all": bench_all,
        "new_all": new_all,
        "ret_data": ret_data,
        "liquid_data": liquid_data,
        "bench_wide": bench_wide,
        "bench_metrics": bench_metrics,
        "tangle_metrics": tangle_metrics,
        "new_metrics": new_metrics,
    }


if __name__ == "__main__":
    run_backtest()
