"""
Ablation experiment configuration
Controls data paths, XGBoost parameters, and switches for various ablation variants.
"""

import os

# ============================================================
# Data paths (requires running on data server)
# ============================================================
FAC_PATH = rf"/project/model_share/share_1/factor_data/fac20250212/fac20250212.fea"
LABEL_PATH = rf"/project/model_share/share_1/label_data/label1.fea"
LIQUID_PATH = rf"/project/model_share/share_1/label_data/can_trade_amt1.fea"

# Use all factors (all columns except date/Code in fac_df), no longer rely on JSON files

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ============================================================
# Backtest quarters (consistent with hhx dynamic3)
# ============================================================
QUARTERS = [
    (2023, 1), (2023, 2), (2023, 3), (2023, 4),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    (2025, 1), (2025, 2),
]

# ============================================================
# XGBoost parameters (identical to hhx dynamic3)
# ============================================================
DEFAULT_XGB_PARAMS = {
    "eta": 0.08,
    "max_depth": 9,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 83,
    "gamma": 1.3,
    "reg_alpha": 0,
    "reg_lambda": 700,
    "objective": "reg:squarederror",
    "disable_default_eval_metric": 1,
    "tree_method": "hist",
    "max_bin": 256,
    "device": "cpu",
}

# ============================================================
# Ray multi-GPU configuration
# ============================================================
RAY_GPU_CONFIG = {
    "enabled": False,          # Set False to fall back to local single-device training
    "num_actors": 4,          # (legacy, kept for interface compat) not used in single-GPU mode
    "cpus_per_actor": 8,      # (legacy, kept for interface compat)
    "gpus_per_actor": 1,      # (legacy, kept for interface compat)
}

# Ray Tune (optional HPO before ablation — disabled by default)
TUNE_CONFIG = {
    "enabled": False,
    "n_trials": 20,
    "num_boost_round": 500,
}

# Training control parameters
TRAIN_CONFIG = {
    "early_stopping_rounds": 50,
    "ndcg_weight": 0.7,
    "rankic_weight": 0.3,
    "ndcg_k": 250,
    "num_boost_round": 20,
}

# Data split parameters
SPLIT_CONFIG = {
    "valid_quarters": 2,   # Validation set span (in quarters)
    "gap_days": 10,        # Gap days between train/validation/test
}

# Evaluation parameters
EVAL_CONFIG = {
    "money": 1.5e9,        # Simulated capital
    "top_k": 200,          # Number of top holdings
}

# hhx baseline reference metrics (from dynamic3 actual measurements)
HHX_BASELINE_METRICS = {
    "IC":       0.1099,
    "ICIR":     1.4695,
    "RankIC":   0.1145,
    "RankICIR": 1.4693,
    "top_return": 0.001999,   # 0.1999% converted to decimal
    "Stability": 0.3890,
}

# ============================================================
# Ablation experiment configuration
# Each configuration modifies only one variable, others remain consistent with baseline
# ============================================================
ABLATION_CONFIGS = {
    # Baseline: fixed 941 factors, original labels, no additional processing
    "baseline": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
    },

    # Method 1: Cross-sectional residual labels (subtract daily market mean)
    "residual_label": {
        "label_mode": "residual",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
    },

    # Method 2: Risk-adjusted labels (divide by individual stock rolling volatility)
    "risk_adjusted_label": {
        "label_mode": "risk_adjusted",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
    },

    # Method 3: Factor IC momentum screening (keep top 70% momentum factors)
    "factor_timing": {
        "label_mode": "raw",
        "factor_timing": True,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
    },

    # Method 4: Industry neutralization (subtract median by A-share industry)
    "industry_neutral": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": True,
        "ensemble_seeds": None,
    },

    # Method 5: Multi-seed ensemble (3 random seeds, validation set IC weighted average)
    "ensemble": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": [42, 100, 200],
    },

    # ============================================================
    # New ablation methods
    # ============================================================

    # Method 6: LambdaMART ranking objective (replace reg:squarederror with rank:pairwise)
    # rank:pairwise uses pairwise logistic loss, works with continuous labels (z-scored)
    # rank:ndcg requires non-negative integer relevance grades, not compatible with z-score
    "lambdamart": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
        "objective_override": "rank:pairwise",
    },

    # Method 7: Temporal sample weighting (exponential decay, recent data weighted higher)
    "temporal_weighting": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
        "temporal_weight_half_life_months": 12,
    },

    # Method 8: Winsorized labels (3-sigma cross-sectional clip before z-score)
    "winsorized_label": {
        "label_mode": "winsorized",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
    },

    # Method 9: Rank-transformed labels (cross-sectional rank/N → uniform [0,1])
    "rank_label": {
        "label_mode": "rank_transform",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
    },

    # Method 10: Rolling window training (cap training data at most recent K quarters)
    "rolling_window": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
        "max_train_quarters": 8,
    },

    # Method 11: Spectral cluster one-hot features (augment factors with cluster membership)
    "spectral_cluster": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
        "spectral_cluster": True,
        "spectral_n_clusters": 8,
        "spectral_n_neighbors": 10,
        "spectral_n_samples": 500,
    },

    # Method 12: Spectral embedding features (augment factors with continuous eigenvector coords)
    "spectral_embedding": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
        "spectral_embedding": True,
        "spectral_n_clusters": 8,
        "spectral_n_neighbors": 10,
        "spectral_n_samples": 500,
    },

    # Method 13: Tangle-weighted ensemble (per-cluster model weights via tangle clustering)
    "tangle_ensemble": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": [42, 100, 200],
        "tangle_ensemble": True,
    },

    # Method 14: Cluster-specific models (train separate XGBoost per spectral cluster)
    "cluster_models": {
        "label_mode": "raw",
        "factor_timing": False,
        "factor_timing_keep_ratio": 0.7,
        "factor_timing_ic_threshold": 0.005,
        "factor_timing_lookback_months": 6,
        "industry_neutral": False,
        "ensemble_seeds": None,
        "cluster_models": True,
        "cluster_n_groups": 4,
        "spectral_n_neighbors": 10,
        "spectral_n_samples": 500,
    },
}
