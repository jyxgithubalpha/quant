from __future__ import annotations

import numpy as np
import polars as pl


def _to_long_liquid(liquid_wide: pl.DataFrame, code_dtype) -> pl.DataFrame:
    liq = liquid_wide.rename({"index": "date"}) if "index" in liquid_wide.columns else liquid_wide
    cols = [c for c in liq.columns if c != "date"]
    out = liq.unpivot(on=cols, index="date", variable_name="Code", value_name="liq")
    if out["Code"].dtype != code_dtype:
        out = out.with_columns(pl.col("Code").cast(code_dtype))
    return out


def simulate_top_return(
    score_df: pl.DataFrame,
    ret_data: pl.DataFrame,
    liquid_wide: pl.DataFrame,
    money: float = 1.5e9,
    top_k: int = 500,
) -> tuple[pl.Series, pl.Series]:
    code_dtype = score_df["Code"].dtype
    ret = ret_data.rename({"label": "ret"})
    if ret["Code"].dtype != code_dtype:
        ret = ret.with_columns(pl.col("Code").cast(code_dtype))
    liq = _to_long_liquid(liquid_wide, code_dtype)

    merged = (
        score_df.join(ret, on=["date", "Code"], how="left")
        .with_columns(pl.col("ret").fill_null(0.0))
        .join(liq, on=["date", "Code"], how="left")
        .with_columns(pl.col("liq").fill_null(0.0))
        .sort(["date", "score"], descending=[False, True])
    )

    rets, ics, dates = [], [], []
    for g in merged.partition_by("date", maintain_order=True):
        dates.append(g["date"][0])
        s = g["score"].to_numpy()
        y = (g["ret"].to_numpy() * 100.0).astype(float)
        l = np.clip(g["liq"].to_numpy().astype(float), 0.0, None)

        n = min(top_k, len(g))
        l_top = l[:n]
        y_top = y[:n]
        cum = np.cumsum(l_top)
        prev = np.concatenate([[0.0], cum[:-1]])
        hold = np.minimum(l_top, np.maximum(0.0, money - prev))
        rets.append(float(np.dot(hold, y_top) / money))

        ics.append(float(np.corrcoef(s, y)[0, 1]))

    df = pl.DataFrame({"date": dates, "ret": rets, "ic": ics})
    return df["ret"], df["ic"]
