from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class DayBatch:
    date: str
    codes: List[str]
    x_alpha: torch.Tensor
    x_style: torch.Tensor
    x_meta: torch.Tensor
    ret_hist: torch.Tensor
    industry: torch.Tensor
    label: torch.Tensor
    liquid: torch.Tensor

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


@dataclass
class Relation:
    name: str
    adj: torch.Tensor
    edge_feat: torch.Tensor


@dataclass
class ForwardOut:
    score: torch.Tensor
    relations: List[Relation] = field(default_factory=list)
    reg_loss: torch.Tensor = field(default_factory=lambda: torch.zeros(()))
