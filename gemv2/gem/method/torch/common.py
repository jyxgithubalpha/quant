"""
PyTorch trainer for tabular regression models (MLP and FT-Transformer).

Supports architecture injection via ``config.params["__arch__"]`` for NNI NAS.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ...core.data import ProcessedBundle
from ...core.training import FitResult, TrainConfig
from ..base.trainer import BaseTrainer


# =============================================================================
# Internal model wrapper
# =============================================================================


class _TorchModelWrapper:
    """Wraps a nn.Module so that predict() works without touching torch at call site."""

    def __init__(self, model: Any, device: Any) -> None:
        self.model = model
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            preds = self.model(x).detach().cpu().numpy()
        return preds


# =============================================================================
# Trainer
# =============================================================================


class TorchTrainer(BaseTrainer):
    """
    Framework trainer for PyTorch tabular regression models.

    Model selection:
    - If ``config.params`` contains ``"__arch__"`` the architecture dict is
      used to build the model via ``_build_from_arch()``.  This path is used
      by ``NNITuner`` for architecture search.
    - Otherwise the ``model_type`` param (default ``"mlp"``) selects between
      the built-in ``FactorMLP`` and ``FTTransformer`` families.

    Key training hyper-parameters (read from ``config.params``):
        lr (float):           Adam learning rate (default 1e-3).
        weight_decay (float): Adam weight decay (default 0.0).
        batch_size (int):     mini-batch size (default 1024).
        use_gpu (bool):       prefer CUDA if available (default False).

    Epoch / patience come from ``config.max_iterations`` and
    ``config.early_stopping_patience`` respectively.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        use_gpu: Optional[bool] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.model_params = model_params or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        views: ProcessedBundle,
        config: TrainConfig,
        phase: str = "full",
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> FitResult:
        import torch

        device = self._resolve_device(config)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        X_train = views.train.X.astype(np.float32)
        y_train = views.train.y.ravel().astype(np.float32)
        X_val = views.val.X.astype(np.float32)
        y_val = views.val.y.ravel().astype(np.float32)

        train_weights = None
        if sample_weights and sample_weights.get("train") is not None:
            train_weights = sample_weights["train"].astype(np.float32)

        params = {**self.model_params, **dict(config.params or {})}
        model = self._build_model(input_dim=X_train.shape[1], params=params)
        model.to(device)

        lr = float(params.get("lr", 1e-3))
        weight_decay = float(params.get("weight_decay", 0.0))
        batch_size = int(params.get("batch_size", 1024))

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        train_loader = self._build_loader(X_train, y_train, train_weights, batch_size, shuffle=True)
        val_loader = self._build_loader(X_val, y_val, None, batch_size, shuffle=False)

        epochs = config.max_iterations
        patience = config.early_stopping_patience

        best_state: Optional[Dict[str, Any]] = None
        best_val = float("inf")
        best_epoch = 0
        val_history: List[float] = []

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb, wb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = torch.mean(wb * (preds - yb) ** 2)
                loss.backward()
                optimizer.step()

            val_loss = self._evaluate_mse(model, val_loader, device)
            val_history.append(val_loss)

            if config.log_interval > 0 and epoch % config.log_interval == 0:
                import logging
                logging.getLogger(__name__).info(
                    "epoch=%d  val_mse=%.6f  best=%.6f", epoch, val_loss, best_val
                )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        wrapper = _TorchModelWrapper(model, device)
        return FitResult(
            model=wrapper,
            evals_result={"val": {"mse": val_history}},
            best_iteration=max(1, best_epoch),
            params=dict(config.params),
            seed=config.seed,
        )

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Dispatch to the wrapper's predict or call directly."""
        if hasattr(model, "predict"):
            return model.predict(X).ravel()
        import torch
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32)
            return model(x).detach().cpu().numpy().ravel()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int, params: Dict[str, Any]) -> Any:
        arch = params.get("__arch__")
        if arch is not None:
            return self._build_from_arch(input_dim, arch)
        return self._build_from_params(input_dim, params)

    def _build_from_params(self, input_dim: int, params: Dict[str, Any]) -> Any:
        from .mlp.model import FactorMLP
        from .ft_transformer.model import FTTransformer

        model_type = str(params.get("model_type", self.model_type)).lower()
        if model_type == "mlp":
            return FactorMLP(
                input_dim=input_dim,
                hidden_sizes=params.get("hidden_sizes"),
                dropout=float(params.get("dropout", 0.0)),
                activation=str(params.get("activation", "relu")),
            )
        if model_type in {"ft_transformer", "ft"}:
            return FTTransformer(
                input_dim=input_dim,
                d_token=int(params.get("d_token", 64)),
                n_heads=int(params.get("n_heads", 4)),
                n_layers=int(params.get("n_layers", 3)),
                dropout=float(params.get("dropout", 0.1)),
                activation=str(params.get("activation", "relu")),
                use_cls_token=bool(params.get("use_cls_token", True)),
            )
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'mlp' or 'ft_transformer'.")

    @staticmethod
    def _build_from_arch(input_dim: int, arch: Dict[str, Any]) -> Any:
        """
        Build a model from an architecture dictionary (e.g. produced by NNI NAS).

        The ``arch`` dict must contain ``"model_type"`` and the corresponding
        model-specific keys.  This dispatches to the same constructors as
        ``_build_from_params`` but skips the model_params merge.
        """
        from .mlp.model import FactorMLP
        from .ft_transformer.model import FTTransformer

        model_type = str(arch.get("model_type", "mlp")).lower()
        if model_type == "mlp":
            return FactorMLP(
                input_dim=input_dim,
                hidden_sizes=arch.get("hidden_sizes"),
                dropout=float(arch.get("dropout", 0.0)),
                activation=str(arch.get("activation", "relu")),
            )
        if model_type in {"ft_transformer", "ft"}:
            return FTTransformer(
                input_dim=input_dim,
                d_token=int(arch.get("d_token", 64)),
                n_heads=int(arch.get("n_heads", 4)),
                n_layers=int(arch.get("n_layers", 3)),
                dropout=float(arch.get("dropout", 0.1)),
                activation=str(arch.get("activation", "relu")),
                use_cls_token=bool(arch.get("use_cls_token", True)),
            )
        raise ValueError(f"Unknown model_type in arch dict: '{model_type}'.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_loader(
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
        batch_size: int,
        shuffle: bool,
    ) -> Any:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        xb = torch.from_numpy(X)
        yb = torch.from_numpy(y)
        wb = torch.from_numpy(weights) if weights is not None else torch.ones(len(X), dtype=torch.float32)
        dataset = TensorDataset(xb, yb, wb)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    @staticmethod
    def _evaluate_mse(model: Any, loader: Any, device: Any) -> float:
        import torch

        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                losses.append(torch.mean((model(xb) - yb) ** 2).item())
        return float(np.mean(losses)) if losses else 0.0

    def _resolve_device(self, config: TrainConfig) -> Any:
        import torch

        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = bool(config.params.get("use_gpu", False))
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


# =============================================================================
# Convenience function
# =============================================================================


def train_torch_model(
    views: ProcessedBundle,
    config: TrainConfig,
    model_type: str = "mlp",
    phase: str = "full",
    sample_weights: Optional[Dict[str, np.ndarray]] = None,
    use_gpu: Optional[bool] = None,
) -> FitResult:
    """
    Convenience wrapper: create a TorchTrainer and call fit() in one step.

    Args:
        views:          processed data bundle.
        config:         training configuration.
        model_type:     ``"mlp"`` or ``"ft_transformer"``.
        phase:          ``"full"`` or ``"tune"``.
        sample_weights: optional per-split weight arrays.
        use_gpu:        override GPU selection; ``None`` defers to config.params.

    Returns:
        FitResult from the trainer.
    """
    trainer = TorchTrainer(model_type=model_type, use_gpu=use_gpu)
    return trainer.fit(views, config, phase=phase, sample_weights=sample_weights)
