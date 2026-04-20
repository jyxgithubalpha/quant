"""Edge-feature-aware message passing per relation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationMessagePassingLayer(nn.Module):
    """Multi-head attention with edge features, one relation."""

    def __init__(self, d_model: int, d_edge_in: int, d_edge: int = 8,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_edge_in = d_edge_in
        self.d_edge = d_edge

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_self = nn.Linear(d_model, d_model)
        self.W_e = nn.Linear(d_edge_in, d_edge * n_heads)
        # per-head logit vector over (q, k, e) concat -> scalar
        self.u = nn.Parameter(torch.randn(n_heads, 2 * self.d_head + d_edge) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, relation) -> torch.Tensor:
        N = h.shape[0]
        H, Dh = self.n_heads, self.d_head

        adj = relation.adj                                         # [N, N]
        edge_feat = relation.edge_feat                             # [N, N, Fe]
        if edge_feat.shape[-1] < self.d_edge_in:
            edge_feat = F.pad(edge_feat, (0, self.d_edge_in - edge_feat.shape[-1]))
        elif edge_feat.shape[-1] > self.d_edge_in:
            edge_feat = edge_feat[..., :self.d_edge_in]

        q = self.W_q(h).view(N, H, Dh)                             # [N, H, Dh]
        k = self.W_k(h).view(N, H, Dh)
        v = self.W_v(h).view(N, H, Dh)
        e = self.W_e(edge_feat).view(N, N, H, self.d_edge)         # [N, N, H, de]

        # Build [N, N, H, 2Dh+de] and dot with u [H, 2Dh+de]
        qi = q.unsqueeze(1).expand(N, N, H, Dh)
        kj = k.unsqueeze(0).expand(N, N, H, Dh)
        feats = torch.cat([qi, kj, e], dim=-1)                     # [N, N, H, 2Dh+de]
        logits = torch.einsum('ijhf,hf->ijh', feats, self.u)       # [N, N, H]

        mask = (adj > 0).unsqueeze(-1)                             # [N, N, 1]
        logits = logits.masked_fill(~mask, float('-inf'))
        # Guard rows with no neighbors -> produce zeros message
        no_nbr = (~mask.any(dim=1, keepdim=True)).expand_as(logits)  # [N, N, H]
        logits = torch.where(no_nbr, torch.zeros_like(logits), logits)

        alpha = F.softmax(logits, dim=1)                           # [N, N, H]
        alpha = alpha * adj.unsqueeze(-1)                          # weight by edge strength
        alpha = self.drop(alpha)

        vj = v.unsqueeze(0).expand(N, N, H, Dh)
        msg = (alpha.unsqueeze(-1) * vj).sum(dim=1).reshape(N, self.d_model)

        out = self.out_proj(msg) + self.W_self(h)
        return self.norm(h + out)


class RelationStack(nn.Module):
    """Per-relation message passing; returns [M, N, D]."""

    def __init__(self, n_relations_max: int, d_model: int, d_edge_in: int = 2,
                 d_edge: int = 8, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RelationMessagePassingLayer(d_model, d_edge_in, d_edge, n_heads, dropout)
                for _ in range(n_layers)
            ])
            for _ in range(n_relations_max)
        ])

    def forward(self, h: torch.Tensor, relations: list) -> torch.Tensor:
        outs = []
        for m, rel in enumerate(relations):
            hm = h
            for layer in self.layers[m]:
                hm = layer(hm, rel)
            outs.append(hm)
        return torch.stack(outs, dim=0)
