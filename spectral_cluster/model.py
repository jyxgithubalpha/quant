"""
model.py — FactorMLP definition, training, and prediction.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class FactorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        # Small init for the final linear layer
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: dict) -> FactorMLP:
    """Train MLP with early stopping, return the best model."""
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = FactorMLP(
        input_dim=X_train.shape[1],
        hidden_dims=config["mlp_hidden"],
        dropout=config["mlp_dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["mlp_lr"])
    criterion = nn.MSELoss()

    train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device)
    val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device)

    bs = config["mlp_batch_size"]
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = len(train_t)

    for epoch in range(config["mlp_epochs"]):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, n_train, bs):
            idx = perm[i: i + bs]
            pred = model(train_t[idx])
            loss = criterion(pred, train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_t)
            val_loss = criterion(val_pred, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"  Epoch {epoch+1:3d}/{config['mlp_epochs']} | "
                  f"Train MSE={avg_train:.6f} | Val MSE={val_loss:.6f} | "
                  f"Patience={patience_counter}/{config['mlp_patience']}")

        if patience_counter >= config["mlp_patience"]:
            print(f"  Early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"  MLP training done, best Val MSE={best_val_loss:.6f}")
    return model


def predict_mlp(model: FactorMLP, X: np.ndarray, device: str = "cuda",
                batch_size: int = 8192) -> np.ndarray:
    """Run MLP inference in batches."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(dev)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i: i + batch_size], dtype=torch.float32, device=dev)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)
