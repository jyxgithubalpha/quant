from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def cross_section_metrics(score_df: pl.DataFrame, ret_data: pl.DataFrame, ndcg_k: int = 200) -> tuple[float, float, float]:
    merged = score_df.join(ret_data, on=["date", "Code"], how="inner")
    rankics, ndcgs = [], []
    for g in merged.partition_by("date", maintain_order=True):
        s = g["score"].to_numpy()
        y = g["label"].to_numpy()
        rankics.append(float(spearmanr(s, y).correlation))
        ndcgs.append(float(ndcg_score([y], [s], k=min(ndcg_k, len(g)))))
    rankic = float(np.nanmean(rankics))
    rankic_std = float(np.nanstd(rankics))
    return rankic, rankic / rankic_std, float(np.nanmean(ndcgs))
