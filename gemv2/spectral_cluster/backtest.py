"""
backtest.py — Backtesting utilities: daily return/IC/RankIC computation,
              overall & quarterly metrics, and delta comparison printing.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def get_ret_ic_rankic(
    score_wide: pd.DataFrame,
    ret_data: pd.DataFrame,
    liquid_data: pd.DataFrame,
    start: str, end: str,
    money: float = 1.5e9, top_k: int = 200,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Daily backtest on a wide score table.
    Returns (daily_ret%, daily_ic, daily_rankic) as pd.Series indexed by date string.

    Parameters
    ----------
    score_wide : index=date (str or datetime), columns=stock_code, values=score
    ret_data   : same layout, label/return wide table
    liquid_data: same layout, liquidity wide table
    start, end : YYYYMMDD date range strings
    """
    # Ensure string index
    if not isinstance(score_wide.index[0], str):
        score_wide = score_wide.copy()
        score_wide.index = score_wide.index.strftime("%Y%m%d")

    eval_dates = [d for d in score_wide.index if start <= d <= end]

    daily_ret, daily_ic, daily_rankic, datelist = [], [], [], []

    for date in eval_dates:
        if date not in ret_data.index or date not in liquid_data.index:
            continue
        datelist.append(date)

        code_rank = score_wide.loc[date].dropna().sort_values(ascending=False)
        if code_rank.empty:
            daily_ret.append(0.0)
            daily_ic.append(np.nan)
            daily_rankic.append(np.nan)
            continue

        ret = ret_data.loc[date].reindex(code_rank.index).fillna(0) * 100
        liquid = liquid_data.loc[date].reindex(code_rank.index).fillna(0)

        # Top-K stock selection return
        total_hold, total_earned = 0.0, 0.0
        for num, code in enumerate(code_rank.index):
            if num >= top_k:
                break
            remain = money - total_hold
            if remain < 1:
                break
            hold = min(remain, liquid[code])
            if hold <= 0:
                continue
            total_hold += hold
            total_earned += ret[code] * hold

        daily_ret.append(total_earned / money if money > 0 else 0.0)

        # Pearson IC
        try:
            daily_ic.append(float(code_rank.corr(ret)))
        except Exception:
            daily_ic.append(np.nan)

        # Rank IC (Spearman)
        try:
            common = code_rank.index.intersection(ret.dropna().index)
            if len(common) >= 5:
                r = spearmanr(code_rank.loc[common].values,
                              ret.loc[common].values).correlation
                daily_rankic.append(float(r) if not np.isnan(r) else np.nan)
            else:
                daily_rankic.append(np.nan)
        except Exception:
            daily_rankic.append(np.nan)

    ret_s = pd.Series(daily_ret, index=datelist, dtype="float")
    ic_s = pd.Series(daily_ic, index=datelist, dtype="float")
    rankic_s = pd.Series(daily_rankic, index=datelist, dtype="float")
    return ret_s, ic_s, rankic_s


def compute_overall_metrics(ret_s: pd.Series, ic_s: pd.Series,
                            rankic_s: pd.Series) -> dict:
    """Compute aggregate backtest metrics."""
    return {
        "IC": float(ic_s.mean()),
        "ICIR": float(ic_s.mean() / ic_s.std()) if ic_s.std() > 0 else 0.0,
        "RankIC": float(rankic_s.mean()),
        "RankICIR": float(rankic_s.mean() / rankic_s.std()) if rankic_s.std() > 0 else 0.0,
        "TopReturn_mean(%)": float(ret_s.mean()),
        "TopReturn_std(%)": float(ret_s.std()),
        "TopReturn_sharpe": float(ret_s.mean() / ret_s.std()) if ret_s.std() > 0 else 0.0,
    }


def compute_quarterly_metrics(ret_s: pd.Series, ic_s: pd.Series,
                              rankic_s: pd.Series) -> pd.DataFrame:
    """Aggregate IC, ICIR, RankIC, TopReturn, Sharpe by quarter."""
    ret_dt = ret_s.copy()
    ic_dt = ic_s.copy()
    ric_dt = rankic_s.copy()
    ret_dt.index = pd.to_datetime(ret_dt.index, format="%Y%m%d")
    ic_dt.index = pd.to_datetime(ic_dt.index, format="%Y%m%d")
    ric_dt.index = pd.to_datetime(ric_dt.index, format="%Y%m%d")

    q_ret_mean = ret_dt.groupby(pd.Grouper(freq="QE")).mean()
    q_ret_std = ret_dt.groupby(pd.Grouper(freq="QE")).std()
    q_ic_mean = ic_dt.groupby(pd.Grouper(freq="QE")).mean()
    q_ic_std = ic_dt.groupby(pd.Grouper(freq="QE")).std()
    q_ric_mean = ric_dt.groupby(pd.Grouper(freq="QE")).mean()

    q_labels = [f"{idx.year}Q{idx.quarter}" for idx in q_ret_mean.index]

    df = pd.DataFrame({
        "IC": q_ic_mean.values,
        "ICIR": (q_ic_mean / q_ic_std.replace(0, np.nan)).values,
        "RankIC": q_ric_mean.values,
        "TopReturn_mean(%)": q_ret_mean.values,
        "TopReturn_std(%)": q_ret_std.values,
        "TopReturn_sharpe": (q_ret_mean / q_ret_std.replace(0, np.nan)).values,
    }, index=q_labels)
    df.index.name = "Quarter"
    return df.round(4)


def print_delta_table(title: str, base_m: dict, target_m: dict,
                      base_name: str = "Base", target_name: str = "Target"):
    """Print absolute + relative delta between two metric dicts."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{'Metric':<22} {base_name:>12} {target_name:>12} {'Delta':>10} {'Rel%':>8}")
    print("-" * 80)
    for k in base_m:
        bv = base_m[k]
        tv = target_m[k]
        delta = tv - bv
        rel = (delta / abs(bv) * 100) if abs(bv) > 1e-8 else 0.0
        sign = "+" if delta >= 0 else ""
        print(f"{k:<22} {bv:>12.4f} {tv:>12.4f} {sign}{delta:>9.4f} {sign}{rel:>7.1f}%")
    print("=" * 80)


def print_metrics_table(all_metrics: dict[str, dict], highlight: str = None):
    """Print a formatted comparison table of multiple methods."""
    print("\n" + "=" * 105)
    print(f"{'Method':<20} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'RankICIR':>8} "
          f"{'TopRet%':>10} {'Sharpe':>8}")
    print("-" * 105)
    for name, m in all_metrics.items():
        marker = " *" if name == highlight else ""
        print(f"{name:<20} {m['IC']:>8.4f} {m['ICIR']:>8.4f} {m['RankIC']:>8.4f} "
              f"{m['RankICIR']:>8.4f} {m['TopReturn_mean(%)']:>10.4f} "
              f"{m['TopReturn_sharpe']:>8.4f}{marker}")
    print("=" * 105)
