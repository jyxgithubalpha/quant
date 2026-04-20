from __future__ import annotations

import polars as pl

from domain.config import DataConfig


def read_ipc_normalized(path: str) -> pl.DataFrame:
    df = pl.read_ipc(path, memory_map=False)
    for col in ("date", "index"):
        if col in df.columns:
            if df[col].dtype == pl.String:
                df = df.with_columns(pl.col(col).str.strptime(pl.Datetime("us"), "%Y%m%d"))
            elif df[col].dtype != pl.Datetime("us"):
                df = df.with_columns(pl.col(col).cast(pl.Datetime("us")))
    if "Code" in df.columns and df["Code"].dtype == pl.String:
        df = df.with_columns(pl.col("Code").cast(pl.Categorical))
    return df


def wide_to_long(wide: pl.DataFrame, value_name: str) -> pl.DataFrame:
    if "index" in wide.columns:
        wide = wide.rename({"index": "date"})
    codes = [c for c in wide.columns if c != "date"]
    return wide.unpivot(on=codes, index="date", variable_name="Code", value_name=value_name)


def load_raw_tables(cfg: DataConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fac = read_ipc_normalized(cfg.fac_path)
    label = read_ipc_normalized(cfg.label_path)
    liquid = read_ipc_normalized(cfg.liquid_path)
    return fac, label, liquid
