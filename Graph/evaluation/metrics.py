from __future__ import annotations

import polars as pl

from .cross_section import cross_section_metrics
from .portfolio import simulate_top_return


def get_metrics(score_df: pl.DataFrame, ret_data: pl.DataFrame, liquid_wide: pl.DataFrame, money: float = 1.5e9) -> dict:
    ret_s, ic_s = simulate_top_return(score_df, ret_data, liquid_wide, money=money)
    rankic, rankicir, ndcg200 = cross_section_metrics(score_df, ret_data)

    ic_mean = float(ic_s.mean())
    ic_std = float(ic_s.std())
    ret_mean = float(ret_s.mean())
    ret_std = float(ret_s.std())

    return {
        "IC": ic_mean,
        "ICIR": ic_mean / ic_std,
        "RankIC": rankic,
        "RankICIR": rankicir,
        "NDCG@200": ndcg200,
        "top_return": ret_mean,
        "Stability": ret_mean / ret_std,
    }
