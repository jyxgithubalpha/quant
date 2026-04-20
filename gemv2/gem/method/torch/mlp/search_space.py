"""
MLPSearchSpace -- Optuna search space for FactorMLP hyper-parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...base.search_space import BaseSearchSpace


@dataclass
class MLPSearchSpace(BaseSearchSpace):
    """
    Search space for FactorMLP architecture and training hyper-parameters.

    Searched dimensions:
        n_layers:     number of hidden layers.
        hidden_size:  width of each hidden layer (same for all layers).
        dropout:      dropout probability.
        activation:   activation function.
        lr:           Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size:   mini-batch size.
    """

    n_layers: Tuple[int, int] = (2, 6)
    hidden_size: Tuple[int, int] = (64, 512)
    dropout: Tuple[float, float] = (0.0, 0.5)
    activation: List[str] = field(default_factory=lambda: ["relu", "gelu", "silu"])
    lr: Tuple[float, float] = (1e-4, 1e-2)
    weight_decay: Tuple[float, float] = (0.0, 0.1)
    batch_size: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])

    def sample(self, trial: Any) -> Dict[str, Any]:
        n_layers = trial.suggest_int("n_layers", self.n_layers[0], self.n_layers[1])
        hidden_sizes = [
            trial.suggest_int(f"hidden_{i}", self.hidden_size[0], self.hidden_size[1])
            for i in range(n_layers)
        ]
        return {
            "model_type": "mlp",
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("dropout", *self.dropout),
            "activation": trial.suggest_categorical("activation", self.activation),
            "lr": trial.suggest_float("lr", *self.lr, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *self.weight_decay),
            "batch_size": trial.suggest_categorical("batch_size", self.batch_size),
        }

    def to_nni_space(self) -> Dict[str, Any]:
        """Convert to NNI search space dict format."""
        return {
            "n_layers": {"_type": "randint", "_value": [self.n_layers[0], self.n_layers[1] + 1]},
            "hidden_size": {"_type": "randint", "_value": [self.hidden_size[0], self.hidden_size[1] + 1]},
            "dropout": {"_type": "uniform", "_value": list(self.dropout)},
            "activation": {"_type": "choice", "_value": self.activation},
            "lr": {"_type": "loguniform", "_value": list(self.lr)},
            "weight_decay": {"_type": "uniform", "_value": list(self.weight_decay)},
            "batch_size": {"_type": "choice", "_value": self.batch_size},
        }
