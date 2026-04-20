"""
plotting.py — All visualization functions for spectral clustering analysis.

Provides:
  - Diagnostic plots (fig1-5): spectral embedding, cluster profiles, prediction
    by cluster, neighbor label consistency, eigenvalue spectrum.
  - Comparison plots (fig6-9): daily IC, cumulative return, relative improvement,
    summary bar chart.
  - Ablation plots (fig10-13): ablation bars, multi-curve cumret/IC, value-add.
  - Ensemble plots (fig10e-14e): ensemble ablation, ensemble cumret/IC/improvement.
  - Generic helpers to avoid code duplication.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph

from config import FIGURES_DIR, MA_WINDOW


# ========================= Helper =========================

def _monthly_xticks(dates: list) -> tuple[list, list]:
    """Extract monthly tick positions and labels from a date list."""
    seen = set()
    ticks, labels = [], []
    for i, d in enumerate(dates):
        m = str(d)[:6]
        if m not in seen:
            seen.add(m)
            ticks.append(i)
            labels.append(str(d))
    return ticks, labels


def _ensure_dir(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)


# ========================= Fig 1: Spectral Embedding =========================

def plot_spectral_embedding(eigenvectors: np.ndarray, cluster_labels: np.ndarray,
                            y: np.ndarray, n_clusters: int,
                            save_dir: str = FIGURES_DIR):
    """
    Scatter plot using the 1st and 2nd non-trivial Laplacian eigenvectors.
    Left: colored by cluster label.  Right: colored by label value.
    """
    _ensure_dir(save_dir)

    v1 = eigenvectors[:, 1]
    v2 = eigenvectors[:, 2] if eigenvectors.shape[1] > 2 else np.zeros_like(v1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1a: cluster label coloring
    ax = axes[0]
    cmap_disc = plt.cm.get_cmap("tab10", n_clusters)
    for c in range(n_clusters):
        mask = cluster_labels == c
        ax.scatter(v1[mask], v2[mask], s=4, alpha=0.5, label=f"C{c}",
                   color=cmap_disc(c))
    ax.set_xlabel("Eigenvector 1 (1st non-trivial)")
    ax.set_ylabel("Eigenvector 2 (2nd non-trivial)")
    ax.set_title("Spectral Embedding — Cluster Labels")
    ax.legend(markerscale=4, fontsize=7, loc="best")

    # 1b: label value coloring
    ax = axes[1]
    vmin, vmax = np.percentile(y, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(v1, v2, s=4, alpha=0.5, c=y, cmap="RdYlGn", norm=norm)
    ax.set_xlabel("Eigenvector 1 (1st non-trivial)")
    ax.set_ylabel("Eigenvector 2 (2nd non-trivial)")
    ax.set_title("Spectral Embedding — Label Values")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Label (zscore)")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_spectral_embedding.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig1] Spectral embedding saved: {path}")


# ========================= Fig 2: Cluster Label Profile =========================

def plot_cluster_label_profile(cluster_labels: np.ndarray, y: np.ndarray,
                               n_clusters: int, save_dir: str = FIGURES_DIR):
    """
    2a: Box plot of label distribution per cluster (sorted by mean).
    2b: Mean / median / std bar chart.
    2c: Dual-axis chart (sample count + mean label).
    """
    _ensure_dir(save_dir)

    stats = []
    for c in range(n_clusters):
        yc = y[cluster_labels == c]
        stats.append({
            "cluster": c,
            "count": len(yc),
            "mean": np.mean(yc) if len(yc) > 0 else 0.0,
            "median": np.median(yc) if len(yc) > 0 else 0.0,
            "std": np.std(yc) if len(yc) > 0 else 0.0,
        })
    stats_sorted = sorted(stats, key=lambda x: x["mean"])
    order = [s["cluster"] for s in stats_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 2a: box plot
    ax = axes[0]
    data_by_cluster = [y[cluster_labels == c] for c in order]
    bp = ax.boxplot(data_by_cluster, labels=[f"C{c}" for c in order],
                    patch_artist=True, showfliers=False)
    cmap = plt.cm.get_cmap("RdYlGn", n_clusters)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(n_clusters - 1, 1)))
    ax.set_title("Label Distribution by Cluster")
    ax.set_ylabel("Label (zscore)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    # 2b: mean / median / std bars
    ax = axes[1]
    x_pos = np.arange(n_clusters)
    means = [s["mean"] for s in stats_sorted]
    medians = [s["median"] for s in stats_sorted]
    stds = [s["std"] for s in stats_sorted]
    w = 0.25
    ax.bar(x_pos - w, means, width=w, label="Mean", color="steelblue")
    ax.bar(x_pos, medians, width=w, label="Median", color="seagreen")
    ax.bar(x_pos + w, stds, width=w, label="Std", color="salmon")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in order])
    ax.set_title("Label Stats by Cluster (sorted by mean)")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    # 2c: dual-axis (count + mean label)
    ax1 = axes[2]
    counts = [s["count"] for s in stats_sorted]
    ax1.bar(x_pos, counts, color="lightblue", alpha=0.7, label="Sample Count")
    ax1.set_ylabel("Sample Count", color="steelblue")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"C{c}" for c in order])
    ax1.set_title("Cluster Size & Mean Label")

    ax2 = ax1.twinx()
    ax2.plot(x_pos, means, "ro-", linewidth=2, markersize=6, label="Mean Label")
    ax2.set_ylabel("Mean Label", color="red")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig2_cluster_label_profile.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig2] Cluster label profile saved: {path}")


# ========================= Fig 3: Prediction by Cluster =========================

def plot_prediction_by_cluster(cl_test: np.ndarray, test_scores: np.ndarray,
                               y_test: np.ndarray, n_clusters: int,
                               save_dir: str = FIGURES_DIR):
    """
    3a: Pearson IC per cluster.
    3b: Rank IC per cluster.
    3c: Average predicted score vs average true label per cluster.
    """
    _ensure_dir(save_dir)

    ics, rank_ics, avg_scores, avg_labels, cluster_ids = [], [], [], [], []
    for c in range(n_clusters):
        mask = cl_test == c
        if mask.sum() < 10:
            continue
        sc_c = test_scores[mask]
        y_c = y_test[mask]
        cluster_ids.append(c)
        avg_scores.append(np.mean(sc_c))
        avg_labels.append(np.mean(y_c))

        if np.std(sc_c) > 1e-8 and np.std(y_c) > 1e-8:
            ics.append(float(np.corrcoef(sc_c, y_c)[0, 1]))
        else:
            ics.append(0.0)

        try:
            r = spearmanr(sc_c, y_c).correlation
            rank_ics.append(float(r) if not np.isnan(r) else 0.0)
        except Exception:
            rank_ics.append(0.0)

    n_valid = len(cluster_ids)
    x_pos = np.arange(n_valid)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 3a: IC
    ax = axes[0]
    colors_ic = ["green" if v > 0 else "red" for v in ics]
    ax.bar(x_pos, ics, color=colors_ic, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax.set_title("Pearson IC by Cluster")
    ax.set_ylabel("IC")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(ics):
        ax.text(i, v + 0.002 * np.sign(v), f"{v:.3f}", ha="center", fontsize=7)

    # 3b: RankIC
    ax = axes[1]
    colors_ric = ["green" if v > 0 else "red" for v in rank_ics]
    ax.bar(x_pos, rank_ics, color=colors_ric, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax.set_title("Rank IC (Spearman) by Cluster")
    ax.set_ylabel("RankIC")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(rank_ics):
        ax.text(i, v + 0.002 * np.sign(v), f"{v:.3f}", ha="center", fontsize=7)

    # 3c: avg score vs avg label
    ax = axes[2]
    w = 0.35
    ax.bar(x_pos - w / 2, avg_scores, width=w, label="Avg Predicted Score",
           color="steelblue", alpha=0.8)
    ax.bar(x_pos + w / 2, avg_labels, width=w, label="Avg True Label",
           color="darkorange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax.set_title("Avg Score vs Avg Label by Cluster")
    ax.set_ylabel("Value")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig3_prediction_by_cluster.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig3] Prediction by cluster saved: {path}")


# ========================= Fig 4: Neighbor Label Consistency =========================

def plot_neighbor_label_consistency(X: np.ndarray, y: np.ndarray,
                                    n_neighbors: int = 10,
                                    save_dir: str = FIGURES_DIR,
                                    seed: int = 42):
    """
    4a: Per-point kNN neighbor label variance histogram.
    4b: Graph-neighbor variance vs random-neighbor variance comparison.
    """
    _ensure_dir(save_dir)
    n = X.shape[0]
    safe_k = max(2, min(n_neighbors, n - 1))

    print(f"  Building kNN graph (k={safe_k}) for neighbor label consistency ...")
    knn = kneighbors_graph(X, n_neighbors=safe_k, mode="connectivity",
                           include_self=False, n_jobs=4)

    # Graph neighbor label variance
    graph_variances = np.zeros(n)
    for i in range(n):
        neighbor_idx = knn[i].indices
        if len(neighbor_idx) > 0:
            graph_variances[i] = np.var(y[neighbor_idx])

    # Random neighbor label variance
    rng = np.random.RandomState(seed)
    random_variances = np.zeros(n)
    for i in range(n):
        rand_idx = rng.choice(n, size=safe_k, replace=False)
        random_variances[i] = np.var(y[rand_idx])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(graph_variances, bins=60, alpha=0.8, color="steelblue", edgecolor="white")
    ax.axvline(x=np.mean(graph_variances), color="red", linestyle="--",
               label=f"Mean={np.mean(graph_variances):.4f}")
    ax.set_title("Per-Point Neighbor Label Variance (kNN Graph)")
    ax.set_xlabel("Variance of neighbor labels")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    ax.hist(graph_variances, bins=60, alpha=0.6, color="steelblue",
            edgecolor="white",
            label=f"Graph neighbors (mean={np.mean(graph_variances):.4f})")
    ax.hist(random_variances, bins=60, alpha=0.6, color="salmon",
            edgecolor="white",
            label=f"Random neighbors (mean={np.mean(random_variances):.4f})")
    ax.set_title("Graph Neighbors vs Random Neighbors — Label Variance")
    ax.set_xlabel("Variance of neighbor labels")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "fig4_neighbor_label_consistency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig4] Neighbor label consistency saved: {path}")

    # Print stats
    print(f"    Graph neighbor var  mean={np.mean(graph_variances):.4f}, "
          f"median={np.median(graph_variances):.4f}")
    print(f"    Random neighbor var mean={np.mean(random_variances):.4f}, "
          f"median={np.median(random_variances):.4f}")
    ratio = np.mean(graph_variances) / max(np.mean(random_variances), 1e-10)
    print(f"    Ratio (graph/random) = {ratio:.4f}"
          f"  {'<- graph captures label smoothness' if ratio < 0.9 else ''}")


# ========================= Fig 5: Eigenvalue Spectrum =========================

def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, n_clusters: int,
                            save_dir: str = FIGURES_DIR):
    """
    5a: Eigenvalue magnitude vs index.
    5b: Eigengap (lambda_{i+1} - lambda_i) vs index.
    """
    _ensure_dir(save_dir)
    n_eig = len(eigenvalues)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(range(n_eig), eigenvalues, "bo-", markersize=5, linewidth=1.5)
    if n_clusters < n_eig:
        ax.axvline(x=n_clusters, color="red", linestyle="--", linewidth=1.5,
                   label=f"n_clusters={n_clusters}")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Spectrum (smallest eigenvalues of L_sym)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for i in range(min(n_eig, n_clusters + 2)):
        ax.annotate(f"{eigenvalues[i]:.4f}", (i, eigenvalues[i]),
                    textcoords="offset points", xytext=(5, 8), fontsize=7)

    ax = axes[1]
    if n_eig > 1:
        gaps = np.diff(eigenvalues)
        colors = ["red" if i == n_clusters - 1 else "steelblue"
                  for i in range(len(gaps))]
        ax.bar(range(1, len(gaps) + 1), gaps, color=colors, alpha=0.8,
               edgecolor="white")
        if n_clusters - 1 < len(gaps):
            ax.annotate(
                f"Gap at k={n_clusters}\nDelta={gaps[n_clusters - 1]:.4f}",
                (n_clusters, gaps[n_clusters - 1]),
                textcoords="offset points", xytext=(15, 15),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red"),
            )
    ax.set_xlabel("Gap Index (between eigenvalue i and i+1)")
    ax.set_ylabel("Eigengap")
    ax.set_title("Eigengap Analysis")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig5_eigenvalue_spectrum.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig5] Eigenvalue spectrum saved: {path}")

    if n_eig > 1:
        gaps = np.diff(eigenvalues)
        max_gap_idx = np.argmax(gaps) + 1
        print(f"    Largest eigengap between {max_gap_idx} -> {max_gap_idx+1}, "
              f"gap={gaps[max_gap_idx - 1]:.4f}")
        if n_clusters - 1 < len(gaps):
            print(f"    Gap at n_clusters={n_clusters}: {gaps[n_clusters - 1]:.4f}")


# ========================= Generic: Improvement Curve =========================

def plot_improvement_curve(series_a: pd.Series, series_b: pd.Series,
                           title: str, filename: str,
                           save_dir: str = FIGURES_DIR):
    """
    Generic (A - B) cumulative improvement curve with green/red fill.
    series_a is the target method, series_b is the baseline.
    """
    _ensure_dir(save_dir)
    diff = series_a - series_b
    cum_diff = diff.cumsum()
    dates = diff.index.tolist()
    ticks, labels = _monthly_xticks(dates)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.fill_between(range(len(cum_diff)), cum_diff.values, 0,
                    where=cum_diff.values >= 0, color="#2a9d8f", alpha=0.35,
                    interpolate=True)
    ax.fill_between(range(len(cum_diff)), cum_diff.values, 0,
                    where=cum_diff.values < 0, color="#e63946", alpha=0.35,
                    interpolate=True)
    ax.plot(cum_diff.values, color="#264653", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    mean_daily = diff.mean()
    win_rate = (diff > 0).mean() * 100
    ax.set_title(f"{title}  |  Mean={mean_daily:.4f}%/day  WinRate={win_rate:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Improvement (%)")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Improvement curve saved: {path}")


# ========================= Generic: Multi-Curve CumRet =========================

def plot_multi_cumret(series_dict: dict[str, pd.Series],
                      title: str = "Cumulative Return (%)",
                      filename: str = "cumret.png",
                      save_dir: str = FIGURES_DIR,
                      styles: dict[str, dict] = None):
    """
    Plot multiple cumulative return curves overlaid.
    series_dict: {label: pd.Series}
    styles: optional {label: dict(color=..., linewidth=..., linestyle=...)}
    """
    _ensure_dir(save_dir)
    default_colors = ["#e63946", "#457b9d", "#e9c46a", "orange", "green", "purple"]
    default_styles = ["--", "-.", ":"]

    first_key = next(iter(series_dict))
    dates = series_dict[first_key].index.tolist()
    ticks, labels = _monthly_xticks(dates)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (name, s) in enumerate(series_dict.items()):
        st = (styles or {}).get(name, {})
        ax.plot(s.cumsum().values, label=name,
                color=st.get("color", default_colors[i % len(default_colors)]),
                linewidth=st.get("linewidth", 1.5),
                linestyle=st.get("linestyle", "-"))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  CumRet plot saved: {path}")


# ========================= Generic: Multi-Curve IC =========================

def plot_multi_ic(series_dict: dict[str, pd.Series],
                  window: int = MA_WINDOW,
                  title: str = "Daily IC Comparison",
                  filename: str = "daily_ic.png",
                  save_dir: str = FIGURES_DIR,
                  styles: dict[str, dict] = None):
    """Plot multiple IC time series (rolling MA) overlaid."""
    _ensure_dir(save_dir)
    default_colors = ["#e63946", "#457b9d", "#e9c46a", "orange", "green", "purple"]

    first_key = next(iter(series_dict))
    dates = series_dict[first_key].index.tolist()
    ticks, labels = _monthly_xticks(dates)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (name, s) in enumerate(series_dict.items()):
        ma = s.rolling(window, min_periods=1).mean() if window > 1 else s
        st = (styles or {}).get(name, {})
        ax.plot(ma.values, label=name,
                color=st.get("color", default_colors[i % len(default_colors)]),
                linewidth=st.get("linewidth", 1.5),
                linestyle=st.get("linestyle", "-"))
    ax.axhline(0, color="gray", linewidth=0.5)
    full_title = f"{title} ({window}-day MA)" if window > 1 else title
    ax.set_title(full_title, fontsize=13, fontweight="bold")
    ax.set_ylabel("IC")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  IC plot saved: {path}")


# ========================= Fig 6: Daily IC with diff =========================

def plot_daily_ic_with_diff(model_ic: pd.Series, bench_ic: pd.Series,
                            window: int = MA_WINDOW,
                            save_dir: str = FIGURES_DIR):
    """IC rolling MA comparison with an improvement subplot."""
    _ensure_dir(save_dir)

    model_ma = model_ic.rolling(window, min_periods=1).mean() if window > 1 else model_ic
    bench_ma = bench_ic.rolling(window, min_periods=1).mean() if window > 1 else bench_ic

    dates = model_ic.index.tolist()
    ticks, labels = _monthly_xticks(dates)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    ax.plot(model_ma.values, label="Spectral MLP", color="#e63946", linewidth=1.5)
    ax.plot(bench_ma.values, label="Bench Equal", color="#457b9d",
            linewidth=1.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"Daily IC Comparison ({window}-day MA)" if window > 1
                 else "Daily IC Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("IC")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    diff = model_ma - bench_ma
    ax.fill_between(range(len(diff)), diff.values, 0,
                    where=diff.values >= 0, color="#2a9d8f", alpha=0.4,
                    interpolate=True)
    ax.fill_between(range(len(diff)), diff.values, 0,
                    where=diff.values < 0, color="#e63946", alpha=0.4,
                    interpolate=True)
    ax.plot(diff.values, color="#264653", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    win_rate = (diff > 0).mean() * 100
    ax.set_title(f"IC Improvement (Model - Bench)  |  WinRate={win_rate:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("IC Diff")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig6_daily_ic.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig6] Daily IC saved: {path}")


# ========================= Fig 9: Summary Bar Chart =========================

def plot_summary_bars(metrics_dict: dict[str, dict],
                      save_dir: str = FIGURES_DIR,
                      filename: str = "fig9_summary_comparison.png"):
    """Multi-method metric comparison bar chart (IC, ICIR, RankIC, TopRet, Sharpe)."""
    _ensure_dir(save_dir)

    names = list(metrics_dict.keys())
    x = np.arange(len(names))
    metric_keys = ["IC", "ICIR", "RankIC", "TopReturn_mean(%)", "TopReturn_sharpe"]
    display_labels = ["IC", "ICIR", "RankIC", "TopRet(%)", "Sharpe"]
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
    n_metrics = len(metric_keys)
    w = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 2.5), 7))
    for j, (mk, dl) in enumerate(zip(metric_keys, display_labels)):
        vals = [metrics_dict[n].get(mk, 0) for n in names]
        bars = ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=dl,
                      color=colors[j], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Summary bar chart saved: {path}")


# ========================= Fig 10: Ablation Bars (2 methods) =========================

def plot_ablation_bars(plain_metrics: dict, spectral_metrics: dict,
                       save_dir: str = FIGURES_DIR,
                       filename: str = "fig10_ablation_bars.png"):
    """Side-by-side comparison of two methods with delta annotations."""
    _ensure_dir(save_dir)

    metric_keys = ["IC", "ICIR", "RankIC", "RankICIR",
                   "TopReturn_mean(%)", "TopReturn_sharpe"]
    display = ["IC", "ICIR", "RankIC", "RankICIR", "TopRet(%)", "Sharpe"]

    plain_vals = [plain_metrics.get(k, 0) for k in metric_keys]
    spec_vals = [spectral_metrics.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - w / 2, plain_vals, w, label="MLP_plain (no cluster)",
           color="#457b9d", alpha=0.85)
    ax.bar(x + w / 2, spec_vals, w, label="MLP_spectral (+ cluster)",
           color="#e63946", alpha=0.85)

    for i in range(len(metric_keys)):
        pv, sv = plain_vals[i], spec_vals[i]
        ax.text(x[i] - w / 2, pv, f"{pv:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + w / 2, sv, f"{sv:.4f}", ha="center", va="bottom", fontsize=7)
        delta = sv - pv
        pct = (delta / abs(pv) * 100) if abs(pv) > 1e-8 else 0.0
        color = "green" if delta > 0 else "red"
        y_pos = max(pv, sv) + 0.002
        ax.annotate(f"D={delta:+.4f}\n({pct:+.1f}%)",
                    xy=(x[i], y_pos), ha="center", fontsize=7,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=10)
    ax.set_title("Ablation: MLP_plain vs MLP_spectral", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Ablation bars saved: {path}")


# ========================= Fig 10e: Ensemble Ablation Bars (3 methods) =====

def plot_ensemble_ablation_bars(bench_m: dict, bench_plain_m: dict,
                                bench_spec_m: dict,
                                save_dir: str = FIGURES_DIR):
    """Bench_Equal vs Bench+MLP_plain vs Bench+MLP_spectral bar chart."""
    _ensure_dir(save_dir)

    metric_keys = ["IC", "ICIR", "RankIC", "RankICIR",
                   "TopReturn_mean(%)", "TopReturn_sharpe"]
    display = ["IC", "ICIR", "RankIC", "RankICIR", "TopRet(%)", "Sharpe"]

    methods = {
        "Bench_Equal": bench_m,
        "Bench+Plain": bench_plain_m,
        "Bench+Spectral": bench_spec_m,
    }
    colors = ["#e9c46a", "#457b9d", "#e63946"]

    x = np.arange(len(metric_keys))
    n_methods = len(methods)
    w = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(16, 7))
    for j, (mname, mvals) in enumerate(methods.items()):
        vals = [mvals.get(k, 0) for k in metric_keys]
        bars = ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=mname,
                      color=colors[j], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=10)
    ax.set_title("Ensemble Ablation: Bench vs Bench+Plain vs Bench+Spectral",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig10_ensemble_ablation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Fig10] Ensemble ablation bars saved: {path}")
