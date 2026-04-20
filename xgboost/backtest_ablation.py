"""
Ablation experiment main runner (optimized version)

Control variable comparison of 5 methods with baseline:
  baseline / residual_label / risk_adjusted_label / factor_timing / industry_neutral / ensemble

Optimization strategy (quarter-priority loop):
  - Outer loop by quarter, inner loop by configuration
  - Feature MAD standardization only once per split per quarter (prepare_features)
  - Label transformation precomputation (3 modes), per-config fast index lookup + z-score
  - factor_timing indices calculated once per quarter, subsequent numpy column slicing reuse
  - Merge RankIC / NDCG groupby into single scan
  - Vectorized portfolio simulation (numpy cumsum)

How to run:
  cd /home/user165/workspace/quant/xgboost
  python backtest_ablation.py

Note: Data must be run on a server where /project/model_share/fac_hhx/ is accessible.
"""

import os
import sys
import gc
import warnings
import time
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Optimize Ray memory usage for large memory systems
os.environ["RAY_object_spilling_threshold"] = "0.9"  # Only spill at 90% memory usage
os.environ["RAY_memory_monitor_refresh_ms"] = "5000"  # Reduce memory monitoring overhead
os.environ["RAY_verbose_spill_logs"] = "0"  # Disable spill logs

import numpy as np
import polars as pl
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# Ensure current directory is in sys.path (when running directly)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FAC_PATH, LABEL_PATH, LIQUID_PATH,
    RESULTS_DIR, QUARTERS, DEFAULT_XGB_PARAMS, TRAIN_CONFIG,
    SPLIT_CONFIG, EVAL_CONFIG, ABLATION_CONFIGS, HHX_BASELINE_METRICS,
    RAY_GPU_CONFIG, TUNE_CONFIG,
)
from model_core import (
    _read_feather, prepare_features, prepare_labels_from_aligned,
    XGBRankModel, get_metrics,
)

import ray
from ray_trainer import RayXGBRankModel, _compute_groups
from label_engineering import transform_label
from factor_timing import select_features_by_timing
from industry_utils import neutralize_scores
from ensemble_utils import train_ensemble, ensemble_predict, train_tangle_ensemble
from spectral_features import augment_with_cluster_onehot, augment_with_spectral_embedding


# ============================================================
# Data splitting (incremental training mode, consistent with dynamic3)
# ============================================================
def get_quarter_range(year: int, quarter: int):
    start_month = (quarter - 1) * 3 + 1
    start = datetime(year, start_month, 1)
    if quarter == 4:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, quarter * 3 + 1, 1) - timedelta(days=1)
    return start, end


def split_data(fac_df: pl.DataFrame, label_df: pl.DataFrame,
               test_year: int, test_quarter: int,
               valid_quarters: int = 2, gap_days: int = 10,
               max_train_quarters: int = None):
    """
    Incremental training data splitting:
      Training set = earliest data date → gap_days days before validation set start
      Validation set = valid_quarters quarters before test set
      Test set = test_year-Qtest_quarter

    If max_train_quarters is set, cap training start to at most that many quarters
    before validation start (rolling window instead of expanding window).
    """
    test_start, test_end = get_quarter_range(test_year, test_quarter)
    valid_end = test_start - timedelta(days=gap_days + 1)
    valid_start = test_start - relativedelta(months=valid_quarters * 3)

    earliest = fac_df["date"].min()
    train_start = earliest
    train_end = valid_start - timedelta(days=gap_days + 1)

    # Rolling window: cap training data to most recent K quarters
    if max_train_quarters is not None:
        rolling_start = valid_start - relativedelta(months=max_train_quarters * 3)
        if rolling_start > train_start:
            train_start = rolling_start
            print(f"  [rolling_window] Training capped to {max_train_quarters}Q: "
                  f"start={train_start.strftime('%Y-%m-%d')}")

    def _slice_fac(start, end):
        return fac_df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    def _slice_label(start, end):
        return label_df.filter((pl.col("index") >= start) & (pl.col("index") <= end))

    train_fac = _slice_fac(train_start, train_end)
    valid_fac = _slice_fac(valid_start, valid_end)
    test_fac = _slice_fac(test_start, test_end)

    train_label = _slice_label(train_start, train_end)
    valid_label = _slice_label(valid_start, valid_end)
    test_label = _slice_label(test_start, test_end)

    print(f"  Split {test_year}-Q{test_quarter}: "
          f"train={len(train_fac)}, val={len(valid_fac)}, test={len(test_fac)}")

    return (
        (train_fac, train_label),
        (valid_fac, valid_label),
        (test_fac, test_label),
        (test_start, test_end),
    )


# ============================================================
# Aggregate quarterly results
# ============================================================
def aggregate_results(results: list) -> dict:
    """Aggregate metrics from multiple quarters into overall mean."""
    valid = [r for r in results if r is not None]
    if not valid:
        return {}

    metrics_df = pl.DataFrame([r["metrics"] for r in valid])
    agg = {col: float(metrics_df[col].mean()) for col in metrics_df.columns}
    return agg


# ============================================================
# Output comparison table
# ============================================================
def print_comparison_table(results_per_config: dict):
    """Print ablation experiment comparison table, including vs baseline delta rows."""
    header = (
        f"{'Config':<24} {'IC':>8} {'ICIR':>8} {'RankIC':>8} "
        f"{'RankICIR':>10} {'TopReturn':>12} {'Stability':>10}"
    )
    sep = "-" * 82

    print()
    print("=" * 82)
    print("Ablation Experiment Comparison")
    print("=" * 82)
    print(header)
    print(sep)

    # First print hhx baseline (reference value)
    ref = HHX_BASELINE_METRICS
    tr_ref = ref["top_return"]
    print(
        f"{'hhx_baseline(ref)':<24} "
        f"{ref['IC']:>8.4f} {ref['ICIR']:>8.4f} "
        f"{ref['RankIC']:>8.4f} {ref['RankICIR']:>10.4f} "
        f"{tr_ref*100:>11.4f}% {ref['Stability']:>10.4f}"
    )
    print(sep)

    # Print each configuration result
    baseline_agg = results_per_config.get("baseline", {})
    for cfg_name, agg in results_per_config.items():
        if not agg:
            print(f"  {cfg_name}: no data")
            continue
        tr = agg.get("top_return", float("nan"))
        print(
            f"{cfg_name:<24} "
            f"{agg.get('IC', float('nan')):>8.4f} "
            f"{agg.get('ICIR', float('nan')):>8.4f} "
            f"{agg.get('RankIC', float('nan')):>8.4f} "
            f"{agg.get('RankICIR', float('nan')):>10.4f} "
            f"{tr*100:>11.4f}% "
            f"{agg.get('Stability', float('nan')):>10.4f}"
        )

    # vs baseline delta rows
    if baseline_agg:
        print()
        print("vs baseline (Δ):")
        print(sep)
        for cfg_name, agg in results_per_config.items():
            if cfg_name == "baseline" or not agg:
                continue
            delta_ic = agg.get("IC", 0) - baseline_agg.get("IC", 0)
            delta_icir = agg.get("ICIR", 0) - baseline_agg.get("ICIR", 0)
            delta_ric = agg.get("RankIC", 0) - baseline_agg.get("RankIC", 0)
            delta_ricir = agg.get("RankICIR", 0) - baseline_agg.get("RankICIR", 0)
            delta_tr = (agg.get("top_return", 0) - baseline_agg.get("top_return", 0)) * 100
            delta_stab = agg.get("Stability", 0) - baseline_agg.get("Stability", 0)
            sign = lambda x: "+" if x >= 0 else ""
            print(
                f"{cfg_name:<24} "
                f"{sign(delta_ic)}{delta_ic:>7.4f} "
                f"{sign(delta_icir)}{delta_icir:>7.4f} "
                f"{sign(delta_ric)}{delta_ric:>7.4f} "
                f"{sign(delta_ricir)}{delta_ricir:>9.4f} "
                f"{sign(delta_tr)}{delta_tr:>10.4f}% "
                f"{sign(delta_stab)}{delta_stab:>9.4f}"
            )

    print("=" * 82)


# ============================================================
# Save results
# ============================================================
def save_results(results_per_config: dict, quarterly_results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Overall summary CSV
    rows = []
    for cfg_name, agg in results_per_config.items():
        if agg:
            row = {"config": cfg_name, **agg}
            rows.append(row)
    if rows:
        summary_df = pl.DataFrame(rows)
        summary_path = os.path.join(RESULTS_DIR, f"ablation_summary_{ts}.csv")
        summary_df.write_csv(summary_path)
        print(f"\nSaved summary results: {summary_path}")

    # Quarterly detail CSV
    detail_rows = []
    for cfg_name, results in quarterly_results.items():
        for r in results:
            if r is not None:
                row = {
                    "config": cfg_name,
                    "year": r["year"],
                    "quarter": r["quarter"],
                    **r["metrics"],
                }
                detail_rows.append(row)
    if detail_rows:
        detail_df = pl.DataFrame(detail_rows)
        detail_path = os.path.join(RESULTS_DIR, f"ablation_quarterly_{ts}.csv")
        detail_df.write_csv(detail_path)
        print(f"Saved quarterly details: {detail_path}")


# ============================================================
# Data loading and label precomputation
# ============================================================
def _load_and_precompute():
    """Load raw data, precompute label transformations for all modes."""
    print("\n[Data Loading]")
    fac_df = _read_feather(FAC_PATH)
    print(f"  Factor data: {len(fac_df):,} rows, {len(fac_df.columns)} columns")
    label_df_raw = _read_feather(LABEL_PATH)
    print(f"  Label data: {len(label_df_raw):,} rows")
    liquid_df = _read_feather(LIQUID_PATH).fill_null(0)
    print(f"  Liquidity data: {len(liquid_df):,} rows")
    initial_features = [c for c in fac_df.columns if c not in ("date", "Code")]
    print(f"  Total factors: {len(initial_features)}")

    print("\n[Precompute Label Transformations]")
    label_modes = sorted(set(cfg["label_mode"] for cfg in ABLATION_CONFIGS.values()))
    label_cache = {}
    for mode in label_modes:
        wide = transform_label(label_df_raw, mode)
        value_cols = [c for c in wide.columns if c != "index"]
        long_pl = wide.unpivot(
            on=value_cols, index="index", variable_name="Code", value_name="label"
        ).rename({"index": "date"})
        label_cache[mode] = long_pl
        print(f"  {mode}: {len(label_cache[mode]):,} records")

    return fac_df, label_df_raw, liquid_df, initial_features, label_cache


# ============================================================
# Cluster-specific model training (Method D)
# ============================================================
def _train_cluster_models(
    X_train, y_train, dates_train,
    X_val, y_val, dates_val,
    X_test, dates_test, codes_test,
    n_groups=4, n_neighbors=10, n_samples=500,
    xgb_params=None, train_config=None,
):
    """Train separate XGBoost models per spectral cluster, merge predictions."""
    import sys as _sys, os as _os
    _sp_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "spectral_cluster")
    if _sp_dir not in _sys.path:
        _sys.path.insert(0, _sp_dir)
    from clustering import spectral_decompose, assign_clusters

    if xgb_params is None:
        xgb_params = DEFAULT_XGB_PARAMS
    if train_config is None:
        train_config = TRAIN_CONFIG

    # Subsample and cluster
    n_train = X_train.shape[0]
    actual_n = min(n_samples, n_train)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_train, size=actual_n, replace=False)
    X_sample = X_train[sample_idx]

    labels_sample, _, _, _ = spectral_decompose(
        X_sample, n_clusters=n_groups, n_neighbors=n_neighbors, seed=42
    )

    train_labels = assign_clusters(X_sample, labels_sample, X_train)
    val_labels = assign_clusters(X_sample, labels_sample, X_val)
    test_labels = assign_clusters(X_sample, labels_sample, X_test)

    unique_clusters = np.unique(train_labels)
    print(f"  [cluster_models] {len(unique_clusters)} clusters, "
          f"sizes: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

    # Train per-cluster models and merge predictions
    all_preds = np.zeros(len(X_test), dtype=np.float64)

    for c in unique_clusters:
        train_mask = train_labels == c
        val_mask = val_labels == c
        test_mask = test_labels == c

        if train_mask.sum() < 50 or val_mask.sum() < 10:
            print(f"    cluster {c}: too few samples (train={train_mask.sum()}, "
                  f"val={val_mask.sum()}), using global model")
            # Fallback: train on all data for this cluster's test stocks
            model = XGBRankModel(xgb_params)
            model.train(
                X_train, y_train, dates_train,
                X_val, y_val, dates_val,
                early_stopping_rounds=train_config["early_stopping_rounds"],
                ndcg_weight=train_config["ndcg_weight"],
                rankic_weight=train_config["rankic_weight"],
                ndcg_k=train_config["ndcg_k"],
                num_boost_round=train_config["num_boost_round"],
                verbose=False,
            )
            preds = model.predict(X_test[test_mask])
            all_preds[test_mask] = preds
            del model
            continue

        dates_train_c = [dates_train[i] for i in range(len(dates_train)) if train_mask[i]]
        dates_val_c = [dates_val[i] for i in range(len(dates_val)) if val_mask[i]]

        model = XGBRankModel(xgb_params)
        model.train(
            X_train[train_mask], y_train[train_mask], dates_train_c,
            X_val[val_mask], y_val[val_mask], dates_val_c,
            early_stopping_rounds=train_config["early_stopping_rounds"],
            ndcg_weight=train_config["ndcg_weight"],
            rankic_weight=train_config["rankic_weight"],
            ndcg_k=train_config["ndcg_k"],
            num_boost_round=train_config["num_boost_round"],
            verbose=False,
        )

        preds = model.predict(X_test[test_mask])
        all_preds[test_mask] = preds
        print(f"    cluster {c}: train={train_mask.sum()}, test={test_mask.sum()}")
        del model

    return pl.DataFrame({"date": dates_test, "Code": codes_test, "score": all_preds})


# ============================================================
# Per-quarter experiment runner
# ============================================================
def _run_quarter(year, quarter, fac_df, label_df_raw, liquid_df,
                 initial_features, label_cache):
    """Run all ablation configurations for a single quarter. Returns {cfg_name: result | None}."""
    print(f"\n{'='*82}")
    print(f"Quarter: {year}-Q{quarter}")

    # Check if any config needs rolling window — use per-config split later if needed
    # For shared features, use the widest (expanding) window
    (train_fac, train_label_raw), (valid_fac, valid_label_raw), \
        (test_fac, test_label_raw), (test_start, test_end) = split_data(
            fac_df, label_df_raw, year, quarter,
            valid_quarters=SPLIT_CONFIG["valid_quarters"],
            gap_days=SPLIT_CONFIG["gap_days"],
        )

    if len(train_fac) == 0 or len(valid_fac) == 0 or len(test_fac) == 0 or \
       len(train_label_raw) == 0 or len(valid_label_raw) == 0 or len(test_label_raw) == 0:
        print(f"  [WARN] Data empty (train_fac={len(train_fac)}, val_fac={len(valid_fac)}, "
              f"test_fac={len(test_fac)}, train_label={len(train_label_raw)}, "
              f"val_label={len(valid_label_raw)}, test_label={len(test_label_raw)}), skipping")
        return {cfg_name: None for cfg_name in ABLATION_CONFIGS}

    select_cols = ["date", "Code"] + initial_features
    print("  [Features] Standardizing training set...")
    X_train_full, dates_train, codes_train, feat_cols, _ = prepare_features(
        train_fac.select(select_cols), train_label_raw
    )
    print("  [Features] Standardizing validation set...")
    X_val_full, dates_val, codes_val, _, _ = prepare_features(
        valid_fac.select(select_cols), valid_label_raw, feat_cols=feat_cols,
    )
    print("  [Features] Standardizing test set...")
    X_test_full, dates_test, codes_test, _, _ = prepare_features(
        test_fac.select(select_cols), test_label_raw, feat_cols=feat_cols,
    )

    del train_fac, valid_fac, test_fac
    gc.collect()

    if len(X_train_full) == 0 or len(X_val_full) == 0 or len(X_test_full) == 0:
        print("  [WARN] Feature matrix empty, skipping")
        del X_train_full, X_val_full, X_test_full
        gc.collect()
        return {cfg_name: None for cfg_name in ABLATION_CONFIGS}

    needs_timing = any(cfg["factor_timing"] for cfg in ABLATION_CONFIGS.values())
    timing_cfg_params = next(
        (cfg for cfg in ABLATION_CONFIGS.values() if cfg["factor_timing"]), None
    )
    timing_indices = None
    if needs_timing:
        selected_features = select_features_by_timing(
            fac_df, label_df_raw, feat_cols, year, quarter,
            lookback_months=timing_cfg_params["factor_timing_lookback_months"],
            keep_ratio=timing_cfg_params["factor_timing_keep_ratio"],
            ic_threshold=timing_cfg_params["factor_timing_ic_threshold"],
        )
        timing_indices = [i for i, f in enumerate(feat_cols) if f in set(selected_features)]

    quarter_results = {}
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        print(f"\n--- {year}-Q{quarter} | {cfg_name} ---")
        label_source = label_cache[cfg["label_mode"]]
        try:
            y_val = prepare_labels_from_aligned(dates_val, codes_val, label_source)

            # --- Rolling window: re-split and re-prepare features if needed ---
            if cfg.get("max_train_quarters"):
                print(f"  [rolling_window] Re-splitting with max_train_quarters={cfg['max_train_quarters']}")
                (rw_train_fac, rw_train_label), _, _, _ = split_data(
                    fac_df, label_df_raw, year, quarter,
                    valid_quarters=SPLIT_CONFIG["valid_quarters"],
                    gap_days=SPLIT_CONFIG["gap_days"],
                    max_train_quarters=cfg["max_train_quarters"],
                )
                select_cols_rw = ["date", "Code"] + initial_features
                X_train_rw, dates_train_rw, codes_train_rw, _, _ = prepare_features(
                    rw_train_fac.select(select_cols_rw), rw_train_label, feat_cols=feat_cols,
                )
                y_train = prepare_labels_from_aligned(dates_train_rw, codes_train_rw, label_source)
                # Override train arrays for this config
                X_train_full_cfg = X_train_rw
                dates_train_cfg = dates_train_rw
                codes_train_cfg = codes_train_rw
                del rw_train_fac, rw_train_label
            else:
                X_train_full_cfg = X_train_full
                dates_train_cfg = dates_train
                codes_train_cfg = codes_train
                y_train = prepare_labels_from_aligned(dates_train_cfg, codes_train_cfg, label_source)

            # --- Feature selection: factor timing ---
            if cfg["factor_timing"] and timing_indices is not None:
                X_train = X_train_full_cfg[:, timing_indices]
                X_val = X_val_full[:, timing_indices]
                X_test = X_test_full[:, timing_indices]
            else:
                X_train, X_val, X_test = X_train_full_cfg, X_val_full, X_test_full

            # --- Feature augmentation: spectral cluster one-hot ---
            if cfg.get("spectral_cluster"):
                X_train, X_val, X_test = augment_with_cluster_onehot(
                    X_train, X_val, X_test,
                    n_clusters=cfg.get("spectral_n_clusters", 8),
                    n_neighbors=cfg.get("spectral_n_neighbors", 10),
                    n_samples=cfg.get("spectral_n_samples", 500),
                )

            # --- Feature augmentation: spectral embedding ---
            if cfg.get("spectral_embedding"):
                X_train, X_val, X_test = augment_with_spectral_embedding(
                    X_train, X_val, X_test,
                    n_clusters=cfg.get("spectral_n_clusters", 8),
                    n_neighbors=cfg.get("spectral_n_neighbors", 10),
                    n_samples=cfg.get("spectral_n_samples", 500),
                )

            val_label_long = (
                pl.DataFrame({"date": dates_val, "Code": codes_val})
                .join(label_source, on=["date", "Code"], how="left")
                .with_columns(pl.col("label").fill_null(0.0))
            )
            test_label_long = (
                pl.DataFrame({"date": dates_test, "Code": codes_test})
                .join(label_source, on=["date", "Code"], how="left")
                .with_columns(pl.col("label").fill_null(0.0))
            )

            # --- Compute optional training modifiers ---
            # Objective override (LambdaMART)
            train_params = dict(DEFAULT_XGB_PARAMS)
            groups_train_arr, groups_val_arr = None, None
            if cfg.get("objective_override"):
                train_params["objective"] = cfg["objective_override"]
                groups_train_arr = _compute_groups(np.asarray(dates_train_cfg))
                groups_val_arr = _compute_groups(np.asarray(dates_val))
                print(f"  [lambdamart] objective={cfg['objective_override']}, "
                      f"train_groups={len(groups_train_arr)}, val_groups={len(groups_val_arr)}")

            # Temporal sample weighting (exponential decay)
            sample_weight = None
            if cfg.get("temporal_weight_half_life_months"):
                half_life = cfg["temporal_weight_half_life_months"]
                dates_arr = np.array(dates_train_cfg)
                # Convert to float days from most recent date
                if hasattr(dates_arr[0], 'timestamp'):
                    t_float = np.array([d.timestamp() for d in dates_arr])
                else:
                    t_float = dates_arr.astype(np.float64)
                t_max = t_float.max()
                t_range = t_max - t_float.min()
                if t_range > 0:
                    decay_rate = np.log(2) / (half_life * 30.0 * 86400)  # half_life in seconds
                    sample_weight = np.exp(-decay_rate * (t_max - t_float)).astype(np.float32)
                    print(f"  [temporal_weighting] half_life={half_life}mo, "
                          f"weight range=[{sample_weight.min():.4f}, {sample_weight.max():.4f}]")

            # --- Training ---
            if cfg.get("tangle_ensemble") and cfg.get("ensemble_seeds"):
                # Tangle-weighted ensemble path
                df_scores = train_tangle_ensemble(
                    X_train, y_train, dates_train_cfg,
                    X_val, y_val, dates_val,
                    X_test, dates_test, codes_test,
                    val_label_long=val_label_long,
                    seeds=cfg["ensemble_seeds"],
                    xgb_params=train_params,
                    train_config=TRAIN_CONFIG,
                    verbose=False,
                )

            elif cfg.get("cluster_models"):
                # Cluster-specific models path
                df_scores = _train_cluster_models(
                    X_train, y_train, dates_train_cfg,
                    X_val, y_val, dates_val,
                    X_test, dates_test, codes_test,
                    n_groups=cfg.get("cluster_n_groups", 4),
                    n_neighbors=cfg.get("spectral_n_neighbors", 10),
                    n_samples=cfg.get("spectral_n_samples", 500),
                    xgb_params=train_params,
                    train_config=TRAIN_CONFIG,
                )

            elif cfg.get("ensemble_seeds"):
                # Standard IC-weighted ensemble
                models, val_ics = train_ensemble(
                    X_train, y_train, dates_train_cfg,
                    X_val, y_val, dates_val,
                    val_label_long=val_label_long,
                    seeds=cfg["ensemble_seeds"],
                    xgb_params=train_params,
                    train_config=TRAIN_CONFIG,
                    verbose=False,
                )
                df_scores = ensemble_predict(models, val_ics, X_test, dates_test, codes_test)
                del models
            else:
                # Single model training
                if RAY_GPU_CONFIG["enabled"]:
                    model = RayXGBRankModel(
                        train_params,
                        num_actors=RAY_GPU_CONFIG["num_actors"],
                        cpus_per_actor=RAY_GPU_CONFIG["cpus_per_actor"],
                        gpus_per_actor=RAY_GPU_CONFIG["gpus_per_actor"],
                    )
                else:
                    model = XGBRankModel(train_params)
                model.train(
                    X_train, y_train, dates_train_cfg,
                    X_val, y_val, dates_val,
                    early_stopping_rounds=TRAIN_CONFIG["early_stopping_rounds"],
                    ndcg_weight=TRAIN_CONFIG["ndcg_weight"],
                    rankic_weight=TRAIN_CONFIG["rankic_weight"],
                    ndcg_k=TRAIN_CONFIG["ndcg_k"],
                    num_boost_round=TRAIN_CONFIG["num_boost_round"],
                    verbose=True,
                    groups_train=groups_train_arr,
                    groups_val=groups_val_arr,
                    sample_weight=sample_weight,
                )
                df_scores = model.predict(X_test, dates_test, codes_test)
                del model

            gc.collect()

            if cfg["industry_neutral"]:
                df_scores = neutralize_scores(df_scores)

            metrics = get_metrics(
                df_scores, test_label_long, liquid_df,
                start=test_start.strftime("%Y%m%d"),
                end=test_end.strftime("%Y%m%d"),
                money=EVAL_CONFIG["money"],
            )
            print(f"  {year}-Q{quarter} | {cfg_name}: "
                  f"IC={metrics['IC']:.4f}, ICIR={metrics['ICIR']:.4f}, "
                  f"RankIC={metrics['RankIC']:.4f}, RankICIR={metrics['RankICIR']:.4f}, "
                  f"top_return={metrics['top_return']*100:.4f}%")
            quarter_results[cfg_name] = {"year": year, "quarter": quarter, "metrics": metrics}
            del df_scores, test_label_long, val_label_long
            gc.collect()

        except Exception as e:
            print(f"  [ERROR] {year}-Q{quarter} | {cfg_name} failed: {e}")
            import traceback
            traceback.print_exc()
            quarter_results[cfg_name] = None

    del X_train_full, X_val_full, X_test_full
    gc.collect()
    return quarter_results


# ============================================================
# Main program
# ============================================================
def main():
    ray.init()

    fac_df, label_df_raw, liquid_df, initial_features, label_cache = _load_and_precompute()

    # Optional: Ray Tune HPO phase (disabled by default via TUNE_CONFIG)
    if TUNE_CONFIG["enabled"] and RAY_GPU_CONFIG["enabled"]:
        from tune_utils import tune_xgb_params
        print("\n[Ray Tune] Running HPO phase...")
        # Use first quarter's data for tuning (rough proxy)
        first_year, first_quarter = QUARTERS[0]
        _tune_split = split_data(
            fac_df, label_df_raw, first_year, first_quarter,
            valid_quarters=SPLIT_CONFIG["valid_quarters"],
            gap_days=SPLIT_CONFIG["gap_days"],
        )
        (tf, tl), (vf, vl), _, _ = _tune_split
        _Xt, _yt_dates, _yt_codes, _feat_cols, _ = prepare_features(tf, tl)
        _Xv, _yv_dates, _yv_codes, _, _ = prepare_features(vf, vl, feat_cols=_feat_cols)
        from label_engineering import transform_label
        _label_raw = transform_label(label_df_raw, "raw")
        _label_cols = [c for c in _label_raw.columns if c != "index"]
        _label_long = _label_raw.unpivot(
            on=_label_cols, index="index", variable_name="Code", value_name="label"
        ).rename({"index": "date"})
        from model_core import prepare_labels_from_aligned
        _yt = prepare_labels_from_aligned(_yt_dates, _yt_codes, _label_long)
        _yv = prepare_labels_from_aligned(_yv_dates, _yv_codes, _label_long)
        best_params = tune_xgb_params(
            _Xt, _yt, _compute_groups(np.asarray(_yt_dates)),
            _Xv, _yv, _compute_groups(np.asarray(_yv_dates)),
            n_trials=TUNE_CONFIG["n_trials"],
            num_boost_round=TUNE_CONFIG["num_boost_round"],
        )
        print(f"[Ray Tune] Best params: {best_params}")
        DEFAULT_XGB_PARAMS.update(best_params)
        del _Xt, _Xv, _yt, _yv, _tune_split

    results_per_config = {cfg_name: [] for cfg_name in ABLATION_CONFIGS}
    for year, quarter in QUARTERS:
        quarter_results = _run_quarter(
            year, quarter, fac_df, label_df_raw, liquid_df,
            initial_features, label_cache,
        )
        for cfg_name, result in quarter_results.items():
            results_per_config[cfg_name].append(result)

    # ---- Aggregate & print ----
    results_agg = {}
    quarterly_results = {}
    for cfg_name in ABLATION_CONFIGS:
        quarterly_results[cfg_name] = results_per_config[cfg_name]
        results_agg[cfg_name] = aggregate_results(results_per_config[cfg_name])
        agg = results_agg[cfg_name]
        if agg:
            print(f"\n[{cfg_name}] Overall mean: "
                  f"IC={agg.get('IC', 0):.4f}, ICIR={agg.get('ICIR', 0):.4f}, "
                  f"RankIC={agg.get('RankIC', 0):.4f}, RankICIR={agg.get('RankICIR', 0):.4f}, "
                  f"top_return={agg.get('top_return', 0)*100:.4f}%")

    # ---- 8. Output comparison table ----
    print_comparison_table(results_agg)

    # ---- 9. Save results ----
    save_results(results_agg, quarterly_results)

    # ---- 10. Shutdown Ray ----
    if RAY_GPU_CONFIG["enabled"] and ray.is_initialized():
        ray.shutdown()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/3600:.2f} hours")
    print("=" * 82)
    print("Ablation experiment completed!")
    print("=" * 82)


if __name__ == "__main__":
    main()
