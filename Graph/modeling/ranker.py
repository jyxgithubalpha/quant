"""MultiRelationalFactorGraphRanker: full assembly."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from domain.config import ModelConfig
from domain.types import DayBatch, ForwardOut
from modeling.encoders import build_encoders
from graphs import (
    PriorGraphBuilder,
    FactorSimilarityGraphBuilder,
    DynamicBehaviorGraphBuilder,
    LatentGraphLearner,
)
from modeling.propagation import RelationStack
from modeling.composer import RelationalSemiringComposer
from modeling.head import RankingHead
from losses import graph_regularizer


class MultiRelationalFactorGraphRanker(nn.Module):
    """Features -> graphs -> propagation -> semiring compose -> rank head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.style_enc, self.alpha_enc, self.tmp_enc, self.fusion = build_encoders(cfg)

        self.prior_builder = PriorGraphBuilder() if cfg.use_prior else None
        self.sim_builder = FactorSimilarityGraphBuilder(topk=cfg.topk_sim) if cfg.use_sim else None
        self.dyn_builder = DynamicBehaviorGraphBuilder(topk=cfg.topk_dyn) if cfg.use_dynamic else None
        self.latent_learner = LatentGraphLearner(cfg.d_model, topk=cfg.topk_latent) if cfg.use_latent else None

        n_relations_max = 4
        self.stack = RelationStack(
            n_relations_max=n_relations_max,
            d_model=cfg.d_model,
            d_edge_in=2,
            d_edge=cfg.d_edge,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_prop_layers,
            dropout=cfg.dropout,
        )
        self.composer = RelationalSemiringComposer(cfg.d_model, n_relations_max, mode=cfg.composer)
        self.head = RankingHead(cfg.d_model)

    def _encode(self, batch: DayBatch) -> torch.Tensor:
        h_style = self.style_enc(batch.x_style)
        h_alpha = self.alpha_enc(batch.x_alpha)
        h_tmp = self.tmp_enc(batch.ret_hist)
        return self.fusion(h_style, h_alpha, h_tmp, batch.x_meta)

    def _build_relations(self, batch: DayBatch, h: torch.Tensor) -> list:
        relations = []
        if self.prior_builder is not None:
            relations.append(self.prior_builder(batch.industry, batch.x_style))
        if self.sim_builder is not None:
            relations.append(self.sim_builder(batch.x_style))
        if self.dyn_builder is not None:
            relations.append(self.dyn_builder(batch.ret_hist))
        if self.latent_learner is not None:
            prior_bias = next((r for r in relations if r.name == "prior"), None)
            relations.append(self.latent_learner(h, prior_bias=prior_bias))
        return relations

    def forward(
        self,
        batch: DayBatch,
        prev_latent_adj: Optional[torch.Tensor] = None,
        pretrain: bool = False,
    ) -> ForwardOut:
        h = self._encode(batch)

        if pretrain:
            score = self.head(h)
            zero = torch.zeros((), device=h.device)
            return ForwardOut(score=score, relations=[], reg_loss=zero)

        relations = self._build_relations(batch, h)
        if not relations:
            score = self.head(h)
            zero = torch.zeros((), device=h.device)
            return ForwardOut(score=score, relations=[], reg_loss=zero)

        zs = self.stack(h, relations)
        z_comp = self.composer(zs)
        score = self.head(z_comp)
        reg = graph_regularizer(relations, prev_latent_adj=prev_latent_adj)
        return ForwardOut(score=score, relations=relations, reg_loss=reg)
