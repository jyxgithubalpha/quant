from __future__ import annotations

from typing import List, Optional

import numpy as np
import polars as pl
import pytorch_lightning as pl_lit
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from domain.config import ModelConfig, TrainConfig
from domain.types import DayBatch
from losses import rank_ic_loss, weighted_pairwise_rank_loss
from model import MultiRelationalFactorGraphRanker


class GraphRankLit(pl_lit.LightningModule):
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = MultiRelationalFactorGraphRanker(model_cfg)
        self.prev_latent_adj: Optional[torch.Tensor] = None
        self._val_ic: list[float] = []
        self._val_ret: list[float] = []
        self._test_outputs: list = []

    def _move_batch(self, batch: List[DayBatch]) -> List[DayBatch]:
        return [d.to(self.device) for d in batch]

    def _top_return(self, scores: torch.Tensor, labels: torch.Tensor, liquid: torch.Tensor) -> float:
        idx = torch.argsort(scores, descending=True)[:500]
        liq = torch.clamp(liquid[idx], min=0.0)
        ret = labels[idx]
        cum = torch.cumsum(liq, dim=0)
        prev = torch.cat([torch.zeros(1, device=liq.device), cum[:-1]])
        hold = torch.minimum(liq, torch.clamp(1.5e9 - prev, min=0.0))
        return float((hold * ret).sum().item() / 1.5e9)

    def training_step(self, batch, batch_idx):
        pretrain = self.model_cfg.two_stage and self.current_epoch < self.model_cfg.pretrain_epochs
        losses = []
        for day in self._move_batch(batch):
            out = self.model(day, prev_latent_adj=self.prev_latent_adj, pretrain=pretrain)
            if not pretrain:
                latent = next((r for r in out.relations if r.name == "latent"), None)
                self.prev_latent_adj = latent.adj.detach() if latent is not None else None
            rank_l = weighted_pairwise_rank_loss(out.score, day.label)
            ic_l = rank_ic_loss(out.score, day.label)
            losses.append(self.model_cfg.w_rank * rank_l + self.model_cfg.w_ic * ic_l + self.model_cfg.w_reg * out.reg_loss)
        loss = torch.stack(losses).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        for day in self._move_batch(batch):
            out = self.model(day, pretrain=False)
            s, y = out.score, day.label
            ic = float(((s - s.mean()) * (y - y.mean())).mean().item() / (s.std().item() * y.std().item() + 1e-8))
            self._val_ic.append(ic)
            self._val_ret.append(self._top_return(s.detach(), y.detach(), day.liquid.detach()))

    def on_validation_epoch_end(self):
        val_ic = float(np.mean(self._val_ic))
        val_top = float(np.mean(self._val_ret))
        self.log("val_ic", val_ic, prog_bar=True)
        self.log("val_top_ret", val_top, prog_bar=True)
        self.log("val_composite", 50.0 * val_top + val_ic, prog_bar=True)
        self._val_ic.clear()
        self._val_ret.clear()

    def test_step(self, batch, batch_idx):
        for day in self._move_batch(batch):
            out = self.model(day, pretrain=False)
            self._test_outputs.append((day.date, list(day.codes), out.score.detach().cpu().numpy()))

    def on_test_epoch_end(self):
        rows_date, rows_code, rows_score = [], [], []
        for date, codes, scores in self._test_outputs:
            rows_date.extend([date] * len(codes))
            rows_code.extend(codes)
            rows_score.extend(scores.tolist())
        self.test_df = pl.DataFrame({"date": rows_date, "Code": rows_code, "score": rows_score}).with_columns(
            pl.col("date").str.strptime(pl.Datetime("us"), "%Y%m%d")
        )
        self._test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.train_cfg.lr, weight_decay=self.train_cfg.weight_decay)

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_composite", mode="max", patience=self.train_cfg.early_stop_patience),
            ModelCheckpoint(monitor="val_composite", mode="max", save_top_k=1, filename="{epoch}-{val_composite:.4f}"),
        ]
