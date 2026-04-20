"""
config.py — Global paths, default hyperparameters, and constants.
"""
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

# ========================= Data paths =========================
FAC_PATH = r"/project/model_share/share_1/factor_data/fac20250212/fac20250212.fea"
LABEL_PATH = r"/project/model_share/share_1/label_data/label1.fea"
LIQUID_PATH = r"/project/model_share/share_1/label_data/can_trade_amt1.fea"

BENCH_PATHS = {
    "bench1": "/project/model_share_remote/score_file/sub_score/label1/bench1_20250822.fea",
    "bench2": "/project/model_share_remote/score_file/sub_score/label1/bench2_20250822.fea",
    "bench3": "/project/model_share_remote/score_file/sub_score/label1/bench3_20250822.fea",
    "bench4": "/project/model_share_remote/score_file/sub_score/label1/bench4_20250822.fea",
    "bench5": "/project/model_share_remote/score_file/bench5_label1.fea",
    "bench6": "/project/model_share_remote/score_file/label1_bench6.fea",
}

# ========================= Default config =========================
DEFAULT_CONFIG = {
    "n_clusters": 8,
    "N_SAMPLES": 500,               # sample size for spectral clustering
    "train_start": "20210101",
    "train_end": "20221231",
    "val_start": "20230101",
    "val_end": "20230630",
    "test_start": "20230101",
    "test_end": "20250630",
    "mlp_lr": 1e-3,
    "mlp_epochs": 100,
    "mlp_patience": 15,
    "mlp_batch_size": 4096,
    "mlp_hidden": [256, 128, 64],
    "mlp_dropout": 0.1,
    "top_k": 200,
    "money": 1.5e9,
    "device": "cuda",
    "seed": 42,
    "clip_range": 5.0,
    "spectral_n_neighbors": 10,
    "spectral_n_jobs": 4,
}

# ========================= Plot constants =========================
FIGURES_DIR = "figures"
MA_WINDOW = 20


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
