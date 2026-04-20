"""
FTTransformerSearchSpace -- Optuna search space for FTTransformer hyper-parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...base.search_space import BaseSearchSpace


@dataclass
class FTTransformerSearchSpace(BaseSearchSpace):
    """
    Search space for FTTransformer architecture and training hyper-parameters.

    Searched dimensions:
        d_token:      token embedding dimension (rounded to nearest 8).
        n_heads:      number of attention heads.
        n_layers:     number of transformer encoder layers.
        dropout:      dropout probability.
        lr:           Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size:   mini-batch size.
    """

    d_token: Tuple[int, int] = (32, 128)
    n_heads: List[int] = field(default_factory=lambda: [2, 4, 8])
    n_layers: Tuple[int, int] = (2, 6)
    dropout: Tuple[float, float] = (0.0, 0.3)
    lr: Tuple[float, float] = (1e-5, 1e-3)
    weight_decay: Tuple[float, float] = (0.0, 0.1)
    batch_size: List[int] = field(default_factory=lambda: [128, 256, 512])

    def sample(self, trial: Any) -> Dict[str, Any]:
        raw_d_token = trial.suggest_int("d_token", self.d_token[0], self.d_token[1])
        # Round to nearest multiple of 8 so d_token is divisible by all n_heads options.
        d_token = max(8, (raw_d_token // 8) * 8)
        return {
            "model_type": "ft_transformer",
            "d_token": d_token,
            "n_heads": trial.suggest_categorical("n_heads", self.n_heads),
            "n_layers": trial.suggest_int("n_layers", self.n_layers[0], self.n_layers[1]),
            "dropout": trial.suggest_float("dropout", *self.dropout),
            "lr": trial.suggest_float("lr", *self.lr, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *self.weight_decay),
            "batch_size": trial.suggest_categorical("batch_size", self.batch_size),
        }

    def to_nni_space(self) -> Dict[str, Any]:
        """Convert to NNI search space dict format."""
        d_values = list(range(self.d_token[0], self.d_token[1] + 1, 8))
        return {
            "d_token": {"_type": "choice", "_value": d_values},
            "n_heads": {"_type": "choice", "_value": self.n_heads},
            "n_layers": {"_type": "randint", "_value": [self.n_layers[0], self.n_layers[1] + 1]},
            "dropout": {"_type": "uniform", "_value": list(self.dropout)},
            "lr": {"_type": "loguniform", "_value": list(self.lr)},
            "weight_decay": {"_type": "uniform", "_value": list(self.weight_decay)},
            "batch_size": {"_type": "choice", "_value": self.batch_size},
        }
