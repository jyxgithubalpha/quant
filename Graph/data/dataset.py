from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from domain.types import DayBatch
from .preprocess import EPS, industry_id_from_code, mad_standardize_cross_section, precompute_ret_hist


class QuarterDataset(Dataset):
    def __init__(
        self,
        fac_long: pl.DataFrame,
        label_long: pl.DataFrame,
        liquid_long: pl.DataFrame,
        dates: list,
        style_cols: list[str],
        alpha_cols: list[str],
        hist_len: int = 20,
        stage: str = "train",
    ):
        self.dates = list(dates)
        self.style_cols = list(style_cols)
        self.alpha_cols = list(alpha_cols)
        self.hist_len = hist_len
        self.stage = stage

        d0, d1 = min(self.dates), max(self.dates)
        hist_start = d0 - timedelta(days=hist_len * 4)

        self.fac_long = fac_long.filter((pl.col("date") >= d0) & (pl.col("date") <= d1)).with_columns(pl.col("Code").cast(pl.String))
        self.label_long = label_long.filter((pl.col("date") >= hist_start) & (pl.col("date") <= d1)).with_columns(pl.col("Code").cast(pl.String))
        self.liquid_long = liquid_long.filter((pl.col("date") >= d0) & (pl.col("date") <= d1)).with_columns(pl.col("Code").cast(pl.String))
        self.ret_hist_cache = precompute_ret_hist(self.label_long, hist_len)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> DayBatch:
        d = self.dates[idx]
        all_cols = self.style_cols + self.alpha_cols
        day = mad_standardize_cross_section(self.fac_long.filter(pl.col("date") == d).sort("Code"), all_cols)

        codes = day["Code"].to_list()
        x_style = day.select(self.style_cols).to_numpy().astype(np.float32)
        x_alpha = day.select(self.alpha_cols).to_numpy().astype(np.float32)

        lab = self.label_long.filter(pl.col("date") == d).select(["Code", "label"]).rename({"label": "_label"})
        liq = self.liquid_long.filter(pl.col("date") == d).select(["Code", "liq"]).rename({"liq": "_liq"})
        joined = pl.DataFrame({"Code": codes}).join(lab, on="Code", how="left").join(liq, on="Code", how="left")
        label_np = joined["_label"].fill_null(0.0).to_numpy().astype(np.float32)
        liquid_np = joined["_liq"].fill_null(0.0).to_numpy().astype(np.float32)

        hist_codes, hist_mat = self.ret_hist_cache[d]
        code_to_row = {c: i for i, c in enumerate(hist_codes)}
        ret_hist = np.zeros((len(codes), self.hist_len), dtype=np.float32)
        for i, c in enumerate(codes):
            r = code_to_row.get(c)
            if r is not None:
                ret_hist[i] = hist_mat[r]

        x_meta = np.stack([np.log1p(np.clip(liquid_np, 0.0, None)), ret_hist[:, -5:].sum(axis=1)], axis=1).astype(np.float32)
        industry = np.array([industry_id_from_code(c) for c in codes], dtype=np.int64)

        if self.stage == "train":
            m, s = label_np.mean(), label_np.std()
            if s > EPS:
                label_np = (label_np - m) / s

        return DayBatch(
            date=d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d),
            codes=codes,
            x_alpha=torch.from_numpy(np.nan_to_num(x_alpha)),
            x_style=torch.from_numpy(np.nan_to_num(x_style)),
            x_meta=torch.from_numpy(np.nan_to_num(x_meta)),
            ret_hist=torch.from_numpy(np.nan_to_num(ret_hist)),
            industry=torch.from_numpy(industry),
            label=torch.from_numpy(np.nan_to_num(label_np)),
            liquid=torch.from_numpy(np.nan_to_num(liquid_np)),
        )


def make_dataloader(dataset: QuarterDataset, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=lambda b: b, drop_last=False)
