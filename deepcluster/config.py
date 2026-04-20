"""DeepCluster V2 for multi-factor financial data — configuration."""

import os

# ============================================================
# Data paths (same as xgboost)
# ============================================================
FAC_PATH = "/project/model_share/share_1/factor_data/fac20250212/fac20250212.fea"
LABEL_PATH = "/project/model_share/share_1/label_data/label1.fea"
LIQUID_PATH = "/project/model_share/share_1/label_data/can_trade_amt1.fea"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ============================================================
# Backtest quarters
# ============================================================
QUARTERS = [
    (2023, 1), (2023, 2), (2023, 3), (2023, 4),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    (2025, 1), (2025, 2),
]

# ============================================================
# DeepCluster V2 model architecture
# ============================================================
MODEL_CONFIG = {
    "encoder_dims": [512, 256, 128],   # Encoder MLP hidden → embedding dim = 128
    "projection_dim": 64,              # Projection head output dim (clustering space)
    "cluster_ks": [50, 100, 200],      # Multi-clustering: 3 different K values
    "temperature": 0.1,                # Softmax temperature for cluster logits
    "dropout": 0.1,                    # Dropout in encoder
}

# ============================================================
# Training
# ============================================================
TRAIN_CONFIG = {
    "n_epochs": 40,                    # Total training epochs (joint clustering + prediction)
    "batch_size": 4096,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "cluster_loss_weight": 1.0,        # Weight for clustering CE loss
    "predict_loss_weight": 1.0,        # Weight for return prediction MSE loss
    "augment_mask_ratio": 0.15,        # Random feature masking ratio
    "augment_noise_std": 0.05,         # Gaussian noise std for augmentation
    "early_stopping_patience": 8,      # Patience on val RankIC
    "reassign_interval": 1,            # Re-cluster every N epochs
}

# ============================================================
# Data split (same as xgboost)
# ============================================================
SPLIT_CONFIG = {
    "valid_quarters": 2,               # Validation set span (quarters)
    "gap_days": 10,                    # Gap between train/val/test
}

# ============================================================
# Evaluation (tangles_ens compatible)
# ============================================================
EVAL_CONFIG = {
    "money": 1.5e9,                    # Simulated capital (RMB)
    "max_stocks": 500,                 # Max stocks in portfolio
}

# ============================================================
# Device
# ============================================================
DEVICE = "cuda"  # "cuda" or "cpu", auto-fallback in trainer
