import torch
import torch.nn as nn

from domain.types import Relation


class PriorGraphBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, industry: torch.Tensor, x_style: torch.Tensor) -> Relation:
        N = industry.shape[0]
        device = industry.device
        same_ind = (industry[:, None] == industry[None, :]).float()

        style_col = x_style[:, 0].contiguous()
        q = torch.quantile(style_col, torch.tensor([1.0 / 3, 2.0 / 3], device=device))
        bucket = torch.bucketize(style_col, q)
        same_bucket = (bucket[:, None] == bucket[None, :]).float()

        eye = torch.eye(N, device=device)
        same_ind = same_ind * (1 - eye)
        same_bucket = same_bucket * (1 - eye)

        adj = torch.where(same_ind > 0, torch.ones_like(same_ind), 0.2 * same_bucket)
        edge_mask = (adj > 0).float().unsqueeze(-1)
        edge_feat = torch.stack([same_ind, same_bucket], dim=-1) * edge_mask
        return Relation(name="prior", adj=adj, edge_feat=edge_feat)
