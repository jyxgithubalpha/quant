"""Backtesting framework for DeepCluster V2.

Evaluation aligned with tangles_ens:
    - Daily rebalance with liquidity constraint
    - IC, ICIR, top_return, top_return_stability
    - Quarterly metrics breakdown
    - Wide score output for comparison
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from tqdm import tqdm

from config import EVAL_CONFIG, QUARTERS, RESULTS_DIR
from data_loader import (
    load_factors, load_labels, load_labels_wide,
    load_liquidity_wide, prepare_quarter_data, quarter_to_dt,
)
from trainer import predict, train_model


# ── Evaluation utilities (tangles_ens compatible) ──────────────────────


def get_ret_ic(
    score: pd.DataFrame,
    ret_data: pd.DataFrame,
    liquid_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    money: float = EVAL_CONFIG["money"],
    max_stocks: int = EVAL_CONFIG["max_stocks"],
    print_quarterly: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Daily rebalance portfolio return and IC with liquidity constraint.

    Args:
        score: wide (index=date_str, columns=code, values=score).
        ret_data: wide daily returns.
        liquid_data: wide tradeable amount per stock.
    """
    idx_range = score.index
    if start is not None:
        idx_range = idx_range[idx_range >= start]
    if end is not None:
        idx_range = idx_range[idx_range <= end]

    label_ret = []
    ic = []
    datelist = []

    for date in idx_range:
        if date not in score.index:
            continue
        datelist.append(date)
        code_rank = score.loc[date].dropna().sort_values(ascending=False)
        if date not in ret_data.index:
            label_ret.append(0.0)
            ic.append(np.nan)
            continue
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

    return (
        pd.Series(label_ret, index=datelist, dtype="float"),
        pd.Series(ic, index=datelist, dtype="float"),
    )


def print_quarterly_metrics(ret: pd.Series, ic: pd.Series):
    """Print quarterly average return and IC."""
    ret_dt = ret.copy()
    ic_dt = ic.copy()
    ret_dt.index = pd.to_datetime(ret_dt.index, format="%Y%m%d")
    ic_dt.index = pd.to_datetime(ic_dt.index, format="%Y%m%d")

    q_ret = ret_dt.groupby(pd.Grouper(freq="QE")).mean()
    q_ic = ic_dt.groupby(pd.Grouper(freq="QE")).mean()
    q_ret.index = [f"{i.year}Q{i.quarter}" for i in q_ret.index]
    q_ic.index = [f"{i.year}Q{i.quarter}" for i in q_ic.index]

    df = pd.DataFrame({
        "Avg Return(%)": q_ret.round(4),
        "Avg IC": q_ic.round(4),
    })
    print("\n===== Quarterly Metrics =====")
    print(df)
    print("=============================\n")


def get_metrics(
    score: pd.DataFrame,
    ret_data: pd.DataFrame,
    liquid_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    print_quarterly: bool = True,
) -> dict:
    """Compute IC, ICIR, top_return, top_return_stability."""
    ret_list, ic_list = get_ret_ic(
        score, ret_data, liquid_data, start=start, end=end,
        print_quarterly=print_quarterly,
    )
    ic_mean = float(ic_list.mean())
    ic_std = float(ic_list.std())
    ret_mean = float(ret_list.mean())
    ret_std = float(ret_list.std())
    return {
        "IC": ic_mean,
        "ICIR": ic_mean / ic_std if ic_std > 1e-9 else 0.0,
        "top_return": ret_mean,
        "top_return_stability": ret_mean / ret_std if ret_std > 1e-9 else 0.0,
    }


def print_metrics(metrics: dict, bench: dict | None = None, title: str = "Evaluation"):
    """Print formatted metric comparison."""
    keys = list(metrics.keys())
    df = pd.DataFrame(index=keys)
    df.index.name = "Metric"
    df["Model"] = [metrics[k] for k in keys]
    if bench is not None:
        df["Bench"] = [bench.get(k) for k in keys]
        df["Improve(%)"] = ((df["Model"] - df["Bench"]) / df["Bench"].abs()) * 100
    print(f"\n{title}")
    print(df.round(4))


# ── Rolling quarterly backtest ─────────────────────────────────────────


def run_backtest():
    """Run DeepCluster V2 backtest over all quarters."""
    print("=" * 60)
    print("DeepCluster V2 — Multi-Factor Backtest")
    print("=" * 60)

    print("\nLoading data...")
    fac_df = load_factors()
    label_df = load_labels()
    ret_data = load_labels_wide()
    liquid_data = load_liquidity_wide()

    feat_cols = [c for c in fac_df.columns if c not in ("date", "Code")]
    print(f"Factors: {len(feat_cols)} features")
    print(f"Factor date range: {fac_df['date'].min()} ~ {fac_df['date'].max()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Collect predictions from all test quarters
    all_records = []  # (date_str, code, score)

    for qi in tqdm(range(len(QUARTERS)), desc="Quarters"):
        year, q = QUARTERS[qi]
        print(f"\n{'=' * 40}")
        print(f"Quarter {year}Q{q}")
        print(f"{'=' * 40}")

        data = prepare_quarter_data(fac_df, label_df, qi)

        if data["X_train"].shape[0] < 1000 or data["X_test"].shape[0] < 100:
            print(f"  Skipping: insufficient data")
            continue

        # Fill NaN in features with 0
        for key in ("X_train", "X_val", "X_test"):
            data[key] = np.nan_to_num(data[key], nan=0.0)

        # Train DeepCluster V2
        model = train_model(
            X_train=data["X_train"],
            y_train=data["y_train"],
            dates_train=data["dates_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            dates_val=data["dates_val"],
            verbose=True,
        )

        # Predict on test quarter
        test_scores = predict(model, data["X_test"])

        # Collect results
        for date, code, score in zip(
            data["dates_test"], data["codes_test"], test_scores,
        ):
            date_str = date.strftime("%Y%m%d") if hasattr(date, "strftime") else str(date)
            all_records.append((date_str, code, float(score)))

    # ── Aggregate results into wide DataFrame ──
    if not all_records:
        print("\nNo predictions generated!")
        return None

    print(f"\n{'=' * 60}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 60}")

    pdf = pd.DataFrame(all_records, columns=["date", "code", "score"])
    wide = pdf.pivot(index="date", columns="code", values="score")

    # Z-score normalize per date (cross-sectional)
    wide = wide.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)
    wide = wide.sort_index()

    print(f"Score matrix: {wide.shape[0]} dates × {wide.shape[1]} stocks")

    # Evaluate
    metrics = get_metrics(wide, ret_data, liquid_data, print_quarterly=True)
    print_metrics(metrics, title="DeepCluster V2 Evaluation")

    # Save results
    wide.to_pickle(os.path.join(RESULTS_DIR, "deepcluster_scores.pkl"))
    print(f"\nScores saved to {RESULTS_DIR}/deepcluster_scores.pkl")

    return {
        "scores_wide": wide,
        "metrics": metrics,
        "ret_data": ret_data,
        "liquid_data": liquid_data,
    }


if __name__ == "__main__":
    run_backtest()
