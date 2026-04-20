"""
FactorMLP -- multi-layer perceptron for cross-sectional factor regression.
"""

from typing import List, Optional

import torch
from torch import nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    cls = activations.get(name, nn.ReLU)
    return cls()


class FactorMLP(nn.Module):
    """
    Fully-connected MLP regressor with optional dropout.

    Args:
        input_dim:    number of input features.
        hidden_sizes: list of hidden layer widths.  Defaults to [256, 128].
        dropout:      dropout probability applied after each hidden activation.
        activation:   activation function name (relu | gelu | silu | tanh).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]
        self.input_dim = input_dim

        layers: List[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def feature_importance(self) -> torch.Tensor:
        """L2 norm of first linear layer weights as a feature importance proxy."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                return torch.norm(layer.weight, dim=0)
        return torch.ones(self.input_dim, device=next(self.parameters()).device)
