"""DeepCluster V2 training loop.

Core algorithm (per epoch):
    1. Forward pass all training data → projected embeddings (no grad)
    2. K-means clustering for each K → pseudo-labels + centroids
    3. Set centroids in model
    4. Mini-batch training: augmented input → cluster CE + prediction MSE → backprop
    5. Evaluate val RankIC → early stopping

Tabular augmentation:
    - Random feature masking (zero out p% of features per sample)
    - Gaussian noise injection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, MODEL_CONFIG, TRAIN_CONFIG
from model import DeepClusterV2


def get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Tabular augmentation ───────────────────────────────────────────────

def augment(x: torch.Tensor, mask_ratio: float, noise_std: float) -> torch.Tensor:
    """Apply random feature masking + Gaussian noise."""
    if mask_ratio > 0:
        mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_ratio))
        x = x * mask
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


# ── Feature extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_projections(
    model: DeepClusterV2,
    X: torch.Tensor,
    batch_size: int = 8192,
) -> np.ndarray:
    """Extract L2-normalized projections for clustering."""
    model.eval()
    parts = []
    for i in range(0, len(X), batch_size):
        z = model.project(X[i:i + batch_size])
        parts.append(z.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── K-means clustering ────────────────────────────────────────────────

def cluster_features(
    projections: np.ndarray,
    cluster_ks: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Run MiniBatchKMeans for each K.

    Returns: {k: (labels, centroids)} for each K.
    """
    result = {}
    for k in cluster_ks:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(4096, len(projections)),
            n_init=3,
            max_iter=100,
            random_state=42,
        )
        labels = kmeans.fit_predict(projections)
        result[k] = (labels, kmeans.cluster_centers_)
    return result


# ── Validation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_rankic(
    model: DeepClusterV2,
    X: torch.Tensor,
    y: np.ndarray,
    dates: list,
    batch_size: int = 8192,
) -> float:
    """Compute mean cross-sectional RankIC on validation set."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        p = model.predict_score(X[i:i + batch_size])
        preds.append(p.cpu().numpy())
    preds = np.concatenate(preds)

    # Group by date, compute Spearman per date
    date_arr = np.array([d for d in dates])  # datetime objects
    unique_dates = sorted(set(date_arr))
    rics = []
    for d in unique_dates:
        mask = date_arr == d
        if mask.sum() < 5:
            continue
        p, lab = preds[mask], y[mask]
        if len(np.unique(p)) < 2 or len(np.unique(lab)) < 2:
            continue
        c = spearmanr(p, lab).correlation
        if not np.isnan(c):
            rics.append(c)
    return float(np.mean(rics)) if rics else 0.0


# ── Training ───────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dates_val: list,
    verbose: bool = True,
) -> DeepClusterV2:
    """Full DeepCluster V2 training loop.

    Returns trained model on CPU.
    """
    device = get_device()
    cfg = TRAIN_CONFIG
    mcfg = MODEL_CONFIG

    input_dim = X_train.shape[1]
    model = DeepClusterV2(
        input_dim=input_dim,
        encoder_dims=mcfg["encoder_dims"],
        projection_dim=mcfg["projection_dim"],
        cluster_ks=mcfg["cluster_ks"],
        temperature=mcfg["temperature"],
        dropout=mcfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"], eta_min=1e-6,
    )

    # Convert to tensors
    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train).to(device)
    X_va_t = torch.from_numpy(X_val).to(device) if len(X_val) > 0 else None

    # DataLoader with indices (for pseudo-label lookup)
    indices = torch.arange(len(X_train))
    dataset = TensorDataset(X_tr_t, y_tr_t, indices)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    best_val_ic = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(cfg["n_epochs"]):
        # ── Phase 1: Extract projections + K-means ──
        if epoch % cfg["reassign_interval"] == 0:
            projections = extract_projections(model, X_tr_t, batch_size=8192)
            cluster_result = cluster_features(projections, mcfg["cluster_ks"])

            # Set centroids in model and prepare pseudo-label tensors
            pseudo_labels = {}
            for k, (labels, centroids) in cluster_result.items():
                model.set_centroids(k, torch.from_numpy(centroids).to(device))
                pseudo_labels[k] = torch.from_numpy(labels).long().to(device)

        # ── Phase 2: Mini-batch training ──
        model.train()
        epoch_cluster_loss = 0.0
        epoch_pred_loss = 0.0
        n_batches = 0

        for X_batch, y_batch, idx_batch in loader:
            # Augment input
            X_aug = augment(X_batch, cfg["augment_mask_ratio"], cfg["augment_noise_std"])

            _, _, logits, pred = model(X_aug)

            # Clustering loss: average CE across all K values
            cluster_loss = torch.tensor(0.0, device=device)
            for k in mcfg["cluster_ks"]:
                targets = pseudo_labels[k][idx_batch]
                cluster_loss = cluster_loss + F.cross_entropy(logits[k], targets)
            cluster_loss = cluster_loss / len(mcfg["cluster_ks"])

            # Prediction loss: MSE on return prediction
            pred_loss = F.mse_loss(pred, y_batch)

            loss = (cfg["cluster_loss_weight"] * cluster_loss
                    + cfg["predict_loss_weight"] * pred_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_cluster_loss += cluster_loss.item()
            epoch_pred_loss += pred_loss.item()
            n_batches += 1

        scheduler.step()

        # ── Phase 3: Validation ──
        val_ic = 0.0
        if X_va_t is not None and len(X_val) > 0:
            val_ic = evaluate_rankic(model, X_va_t, y_val, dates_val)

        if verbose and (epoch % 5 == 0 or epoch == cfg["n_epochs"] - 1):
            print(f"  Epoch {epoch:3d} | "
                  f"cluster_loss={epoch_cluster_loss / n_batches:.4f} "
                  f"pred_loss={epoch_pred_loss / n_batches:.4f} "
                  f"val_RankIC={val_ic:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping on val RankIC
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}, best val_RankIC={best_val_ic:.4f}")
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    model.eval()

    if verbose:
        print(f"  Training done. Best val_RankIC={best_val_ic:.4f}")
    return model


# ── Inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: DeepClusterV2,
    X: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """Generate return predictions."""
    model.eval()
    device = next(model.parameters()).device
    X_t = torch.from_numpy(X).to(device)
    preds = []
    for i in range(0, len(X_t), batch_size):
        p = model.predict_score(X_t[i:i + batch_size])
        preds.append(p.cpu().numpy())
    return np.concatenate(preds)
