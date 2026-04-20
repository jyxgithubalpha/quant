"""Tangle ensemble configuration."""

BENCH_PATHS = {
    "bench1": "/project/model_share_remote/score_file/sub_score/label1/bench1_20250822.fea",
    "bench2": "/project/model_share_remote/score_file/sub_score/label1/bench2_20250822.fea",
    "bench3": "/project/model_share_remote/score_file/sub_score/label1/bench3_20250822.fea",
    "bench4": "/project/model_share_remote/score_file/sub_score/label1/bench4_20250822.fea",
    "bench5": "/project/model_share_remote/score_file/bench5_label1.fea",
    "bench6": "/project/model_share_remote/score_file/label1_bench6.fea",
}

LABEL_PATH = "/project/model_share/share_1/label_data/label1.fea"
LIQUID_PATH = "/project/model_share/share_1/label_data/can_trade_amt1.fea"

# Top portfolio
PORTFOLIO_CAPITAL = 1.5e9  # fixed capital for portfolio simulation (RMB)
MAX_STOCKS = 500           # max stocks to consider

# Tangle parameters
AGREEMENT_RATIO = 0.05   # agreement = int(n_stocks * ratio)
N_QUANTILES = 4           # quantile splits per bench → cuts at 20%, 40%, 60%, 80%
PRUNE_DEPTH = 1
MIN_CLUSTER_SIZE = 30     # minimum stocks per cluster for valid weight estimation

# Weight calibration
LOOKBACK_DAYS = 60        # rolling window for per-cluster IC-based weights

# Visualization
MA_WINDOW = 1            # moving average window for plots

# Evaluation period
EVAL_START = "20230201"   # skip first month for lookback warmup
EVAL_END = "20250101"
