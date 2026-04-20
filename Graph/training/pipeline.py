from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytorch_lightning as pl_lit

from data import QuarterDataset, build_season_splits, load_raw_tables, make_dataloader, split_style_alpha_cols, wide_to_long
from domain.config import ExperimentConfig, ModelConfig
from .lightning_module import GraphRankLit


def train_one_season(year: int, quarter: int, ablation: str = "baseline", seed: int = 42, exp_cfg: ExperimentConfig | None = None) -> pl.DataFrame:
    exp_cfg = exp_cfg or ExperimentConfig()
    pl_lit.seed_everything(seed, workers=True)

    fac_long, label_wide, liquid_wide = load_raw_tables(exp_cfg.data)
    label_long = wide_to_long(label_wide, "label")
    liquid_long = wide_to_long(liquid_wide, "liq")

    train_dates, valid_dates, test_dates = build_season_splits(
        fac_long,
        year,
        quarter,
        valid_quarters=exp_cfg.split.valid_quarters,
        gap_days=exp_cfg.split.gap_days,
    )

    style_cols, alpha_cols = split_style_alpha_cols(fac_long, (min(train_dates), max(train_dates)), n_style=40)
    model_cfg = replace(ModelConfig(f_alpha=len(alpha_cols), f_style=len(style_cols)), **exp_cfg.ablations.get(ablation, {}))

    train_ds = QuarterDataset(fac_long, label_long, liquid_long, train_dates, style_cols, alpha_cols, model_cfg.hist_len, "train")
    valid_ds = QuarterDataset(fac_long, label_long, liquid_long, valid_dates, style_cols, alpha_cols, model_cfg.hist_len, "valid")
    test_ds = QuarterDataset(fac_long, label_long, liquid_long, test_dates, style_cols, alpha_cols, model_cfg.hist_len, "test")

    train_dl = make_dataloader(train_ds, batch_size=exp_cfg.train.batch_size, shuffle=True)
    valid_dl = make_dataloader(valid_ds, batch_size=1, shuffle=False)
    test_dl = make_dataloader(test_ds, batch_size=1, shuffle=False)

    lit = GraphRankLit(model_cfg=model_cfg, train_cfg=exp_cfg.train)
    trainer = pl_lit.Trainer(
        max_epochs=exp_cfg.train.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, train_dl, valid_dl)
    trainer.test(lit, test_dl, ckpt_path=trainer.checkpoint_callback.best_model_path or None)
    return lit.test_df
