import torch
import torch.nn as nn

from domain.types import Relation
from .similarity import topk_mask


class DynamicBehaviorGraphBuilder(nn.Module):
    def __init__(self, topk: int = 20):
        super().__init__()
        self.topk = topk

    def forward(self, ret_hist: torch.Tensor) -> Relation:
        L = ret_hist.shape[-1]
        x = ret_hist - ret_hist.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-8)
        corr = (x @ x.T) / L
        N = corr.shape[0]
        eye = torch.eye(N, dtype=torch.bool, device=corr.device)
        corr = corr.masked_fill(eye, 0.0)
        mask = topk_mask(corr.abs(), self.topk)
        adj = corr.clamp(min=0.0) * mask.float()
        edge_mask = (adj > 0).float()
        edge_feat = (corr * edge_mask).unsqueeze(-1)
        return Relation(name="dynamic", adj=adj, edge_feat=edge_feat)
