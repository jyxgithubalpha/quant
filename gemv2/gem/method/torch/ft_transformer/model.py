"""
FTTransformer -- Feature Tokenizer + Transformer for tabular regression.

Architecture reference:
    Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).
"""

from typing import Optional

import torch
from torch import nn


class _FeatureTokenizer(nn.Module):
    """Maps each scalar feature to a d_token-dimensional embedding."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)  ->  (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias

    def importance(self) -> torch.Tensor:
        """Per-feature L2 norm of token weights."""
        return torch.norm(self.weight, dim=1)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer regressor.

    Args:
        input_dim:     number of input features.
        d_token:       token embedding dimension (must be divisible by n_heads).
        n_heads:       number of attention heads.
        n_layers:      number of transformer encoder layers.
        dropout:       dropout rate inside TransformerEncoderLayer.
        activation:    feedforward activation (``"relu"`` or ``"gelu"``).
        use_cls_token: if True, prepend a learnable CLS token and use it for
                       the regression head; otherwise use mean pooling.
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = _FeatureTokenizer(input_dim, d_token)
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token: Optional[nn.Parameter] = nn.Parameter(torch.zeros(1, 1, d_token))
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        if self.use_cls_token and self.cls_token is not None:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0] if self.use_cls_token else encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)

    def feature_importance(self) -> torch.Tensor:
        """Per-feature importance from the tokenizer weight norms."""
        return self.tokenizer.importance()
