import torch
import torch.nn as nn
import torch.nn.functional as F

from domain.types import Relation


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    N = scores.shape[0]
    k = min(k, N - 1)
    _, idx = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


class FactorSimilarityGraphBuilder(nn.Module):
    def __init__(self, topk: int = 20):
        super().__init__()
        self.topk = topk

    def forward(self, h_style: torch.Tensor) -> Relation:
        h = F.normalize(h_style, dim=-1, eps=1e-8)
        sim = h @ h.T
        N = sim.shape[0]
        eye = torch.eye(N, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(eye, 0.0)
        mask = topk_mask(sim, self.topk)
        adj = sim.clamp(min=0.0) * mask.float()
        edge_feat = (sim * mask.float()).unsqueeze(-1)
        return Relation(name="factor_sim", adj=adj, edge_feat=edge_feat)
