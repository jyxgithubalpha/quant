"""Output heads."""

import torch
import torch.nn as nn


class RankingHead(nn.Module):
    """z [N, D] -> score [N]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z).squeeze(-1)


class PairwiseHead(nn.Module):
    """Pairwise preference probability helper."""

    def pairwise_logits(self, scores: torch.Tensor) -> torch.Tensor:
        return scores.unsqueeze(1) - scores.unsqueeze(0)
