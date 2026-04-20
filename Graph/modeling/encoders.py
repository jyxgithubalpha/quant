"""Node feature encoders: style / alpha / temporal / fusion."""

from __future__ import annotations

import torch
from torch import nn

from domain.config import ModelConfig


def _kaiming(m: nn.Module) -> None:
    """Kaiming-uniform init for all Linear layers, zero bias."""
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, mode="fan_in", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)
        elif isinstance(mod, nn.Conv1d):
            nn.init.kaiming_uniform_(mod.weight, mode="fan_in", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)


class StyleEncoder(nn.Module):
    """[N, F_style] -> [N, D_style] via 2-layer MLP with LeakyReLU + Dropout."""

    def __init__(self, f_style: int, d_style: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_style * 2, 32)
        self.net = nn.Sequential(
            nn.Linear(f_style, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_style),
        )
        _kaiming(self)

    def forward(self, x_style: torch.Tensor) -> torch.Tensor:
        return self.net(x_style)


class AlphaEncoder(nn.Module):
    """[N, F_alpha] -> [N, D_alpha] with wide bottleneck + residual MLP."""

    def __init__(self, f_alpha: int, d_alpha: int, dropout: float = 0.2):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Linear(f_alpha, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )
        self.block = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
        )
        self.head = nn.Linear(128, d_alpha)
        _kaiming(self)

    def forward(self, x_alpha: torch.Tensor) -> torch.Tensor:
        h = self.filter(x_alpha)
        h = h + self.block(h)
        return self.head(h)


class _TCNBlock(nn.Module):
    """Dilated Conv1d -> GELU -> Dropout, residual."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.conv(x)))


class TCNTemporalEncoder(nn.Module):
    """[N, L] -> [N, D_tmp] via dilated 1D TCN, last-step pool."""

    def __init__(self, hist_len: int, d_tmp: int, dropout: float = 0.1, channels: int = 32):
        super().__init__()
        self.proj_in = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList([_TCNBlock(channels, d, dropout) for d in (1, 2, 4)])
        self.proj_out = nn.Linear(channels, d_tmp)
        _kaiming(self)

    def forward(self, ret_hist: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(ret_hist.unsqueeze(1))
        for blk in self.blocks:
            h = blk(h)
        return self.proj_out(h[:, :, -1])


class GRUTemporalEncoder(nn.Module):
    """[N, L] -> [N, D_tmp] via 1-layer GRU, last hidden."""

    def __init__(self, hist_len: int, d_tmp: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=d_tmp, num_layers=1, batch_first=True)

    def forward(self, ret_hist: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(ret_hist.unsqueeze(-1))
        return h.squeeze(0)


class NodeFeatureFusion(nn.Module):
    """Concat [h_style, h_alpha, h_tmp, x_meta] -> D_model via MLP + LayerNorm."""

    def __init__(self, d_style: int, d_alpha: int, d_tmp: int, f_meta: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_in = d_style + d_alpha + d_tmp + f_meta
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_model * 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        _kaiming(self)

    def forward(self, h_style, h_alpha, h_tmp, x_meta) -> torch.Tensor:
        h = torch.cat([h_style, h_alpha, h_tmp, x_meta], dim=-1)
        return self.norm(self.mlp(h))


def build_encoders(cfg: ModelConfig) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Factory: (style_enc, alpha_enc, tmp_enc, fusion), tmp type from cfg.temporal_encoder."""
    style = StyleEncoder(cfg.f_style, cfg.d_style, cfg.dropout)
    alpha = AlphaEncoder(cfg.f_alpha, cfg.d_alpha, cfg.dropout)
    if cfg.temporal_encoder == "tcn":
        tmp: nn.Module = TCNTemporalEncoder(cfg.hist_len, cfg.d_tmp, cfg.dropout)
    elif cfg.temporal_encoder == "gru":
        tmp = GRUTemporalEncoder(cfg.hist_len, cfg.d_tmp, cfg.dropout)
    else:
        raise ValueError(f"unknown temporal_encoder: {cfg.temporal_encoder}")
    fusion = NodeFeatureFusion(cfg.d_style, cfg.d_alpha, cfg.d_tmp, cfg.f_meta, cfg.d_model, cfg.dropout)
    return style, alpha, tmp, fusion
