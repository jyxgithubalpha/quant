"""
run_basic.py — Basic spectral clustering + MLP pipeline with backtest.
(Replaces the original main.py)
"""
import os
import gc
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime

from config import (FAC_PATH, LABEL_PATH, LIQUID_PATH, DEFAULT_CONFIG,
                    FIGURES_DIR, set_seed)
from data import prepare_data, split_by_date, subsample
from clustering import spectral_decompose, assign_clusters, build_features_with_cluster
from model import train_mlp, predict_mlp
from backtest import get_ret_ic_rankic, compute_overall_metrics
from plotting import (plot_spectral_embedding, plot_cluster_label_profile,
                      plot_eigenvalue_spectrum)


def main(config: dict = None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    set_seed(cfg["seed"])

    save_dir = FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("Spectral Clustering + MLP Factor Analysis & Backtest")
    print("=" * 80)
    print(f"N_SAMPLES (spectral): {cfg['N_SAMPLES']}")
    print(f"n_clusters: {cfg['n_clusters']}")
    print(f"Test period: {cfg['test_start']} ~ {cfg['test_end']}")
    print()

    # ---------- Load data ----------
    print("Loading data ...")
    fac_df = pl.read_ipc(FAC_PATH, memory_map=False)
    label_df = pl.read_ipc(LABEL_PATH, memory_map=False)
    liquid_df = pl.read_ipc(LIQUID_PATH, memory_map=False)
    print(f"  fac: {fac_df.shape}, label: {label_df.shape}, liquid: {liquid_df.shape}")

    # ---------- Prepare ----------
    print("\nData preparation (cross-sectional normalization) ...")
    all_df, fac_cols = prepare_data(fac_df, label_df, liquid_df,
                                    clip_range=cfg["clip_range"])

    train_df = split_by_date(all_df, cfg["train_start"], cfg["train_end"])
    val_df = split_by_date(all_df, cfg["val_start"], cfg["val_end"])
    test_df = split_by_date(all_df, cfg["test_start"], cfg["test_end"])
    print(f"  Train: {train_df.shape[0]}, Val: {val_df.shape[0]}, Test: {test_df.shape[0]}")

    X_train = train_df.select(fac_cols).to_numpy().astype(np.float32)
    y_train = train_df["label"].to_numpy().astype(np.float32)
    X_val = val_df.select(fac_cols).to_numpy().astype(np.float32)
    y_val = val_df["label"].to_numpy().astype(np.float32)
    X_test = test_df.select(fac_cols).to_numpy().astype(np.float32)

    # ---------- Spectral clustering (sampled) ----------
    print(f"\nSpectral decomposition (sampling {cfg['N_SAMPLES']}) ...")
    sample_train = subsample(train_df, cfg["N_SAMPLES"], seed=cfg["seed"])
    X_sample = sample_train.select(fac_cols).to_numpy().astype(np.float32)
    y_sample = sample_train["label"].to_numpy().astype(np.float32)

    cl_sample, eigenvalues, eigenvectors, _ = spectral_decompose(
        X_sample, n_clusters=cfg["n_clusters"],
        n_neighbors=cfg["spectral_n_neighbors"],
        n_jobs=cfg["spectral_n_jobs"], seed=cfg["seed"],
    )

    # Diagnostic plots
    plot_eigenvalue_spectrum(eigenvalues, cfg["n_clusters"], save_dir=save_dir)
    plot_spectral_embedding(eigenvectors, cl_sample, y_sample,
                            cfg["n_clusters"], save_dir=save_dir)
    plot_cluster_label_profile(cl_sample, y_sample,
                               cfg["n_clusters"], save_dir=save_dir)

    # Propagate clusters
    print("Assigning clusters to train / val / test ...")
    cl_train = assign_clusters(X_sample, cl_sample, X_train)
    cl_val = assign_clusters(X_sample, cl_sample, X_val)
    cl_test = assign_clusters(X_sample, cl_sample, X_test)

    del X_sample, y_sample, sample_train
    gc.collect()

    # ---------- Build MLP features ----------
    n_cl = cfg["n_clusters"]
    X_train_mlp = build_features_with_cluster(X_train, cl_train, n_cl)
    X_val_mlp = build_features_with_cluster(X_val, cl_val, n_cl)
    X_test_mlp = build_features_with_cluster(X_test, cl_test, n_cl)
    print(f"MLP input dim: {X_train_mlp.shape[1]} "
          f"(factors {len(fac_cols)} + cluster {n_cl})")

    # ---------- Train MLP ----------
    print("\nMLP training ...")
    model = train_mlp(X_train_mlp, y_train, X_val_mlp, y_val, cfg)

    # ---------- Predict ----------
    print("\nGenerating predictions ...")
    test_scores = predict_mlp(model, X_test_mlp, device=cfg["device"])

    score_df = test_df.select(["date", "Code"]).with_columns(
        pl.Series("score", test_scores)
    )

    # ---------- Backtest ----------
    print(f"\nBacktest ({cfg['test_start']} ~ {cfg['test_end']}) ...")
    score_pd = score_df.to_pandas()
    score_pd["date"] = score_pd["date"].astype(str).str.replace("-", "")
    model_wide = score_pd.pivot(index="date", columns="Code",
                                values="score").sort_index()

    ret_data = pd.read_feather(LABEL_PATH).set_index("index")
    ret_data.index = ret_data.index.astype(str)
    liquid_data = pd.read_feather(LIQUID_PATH).set_index("index")
    liquid_data.index = liquid_data.index.astype(str)

    model_ret, model_ic, model_ric = get_ret_ic_rankic(
        model_wide, ret_data, liquid_data,
        start=cfg["test_start"], end=cfg["test_end"],
        money=cfg["money"], top_k=cfg["top_k"],
    )
    metrics = compute_overall_metrics(model_ret, model_ic, model_ric)

    print("\n" + "=" * 60)
    print("Backtest Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save scores
    os.makedirs("scores", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"scores/spectral_mlp_{ts}.csv"
    score_df.write_csv(csv_path)
    print(f"\nScore table saved: {csv_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return metrics, score_df


if __name__ == "__main__":
    main()
