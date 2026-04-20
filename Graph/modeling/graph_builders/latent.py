import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from domain.types import Relation


class LatentGraphLearner(nn.Module):
    def __init__(self, d_model: int, topk: int = 20):
        super().__init__()
        self.d_model = d_model
        self.topk = topk
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.prior_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, prior_bias: Optional[Relation] = None) -> Relation:
        Q = self.W_q(h)
        K = self.W_k(h)
        logits = (Q @ K.T) / math.sqrt(self.d_model)

        if prior_bias is not None:
            logits = logits + self.prior_scale * prior_bias.adj

        N = logits.shape[0]
        eye = torch.eye(N, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye, float("-inf"))

        k = min(self.topk, N - 1)
        _, idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, idx, True)

        masked_logits = logits.masked_fill(~mask, float("-inf"))
        attn = F.softmax(masked_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        return Relation(name="latent", adj=attn, edge_feat=attn.unsqueeze(-1))
