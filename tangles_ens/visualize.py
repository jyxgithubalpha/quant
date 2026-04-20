"""Visualize tangle ensemble vs bench baseline comparison.

Aligned with bench notebook style:
- Cumulative return comparison (model vs bench)
- Relative improvement plot
- Tangle tree structure (diagnostic)
- Cluster weight heatmap (diagnostic)
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from config import (
    BENCH_PATHS, EVAL_START, EVAL_END, LOOKBACK_DAYS,
    MA_WINDOW, LABEL_PATH, LIQUID_PATH,
)
from data_loader import load_data
from backtest import (
    get_ret_ic, get_metrics, ensemble_scores,
    load_wide_pandas, compute_tangle_scores_wide,
    BENCH_COLS,
)
from tangle_ensemble import run_tangle_full, compute_cluster_weights

OUTPUT_DIR = Path("plots")


# ── 1. Cumulative Return (bench notebook style) ─────────────────────────


def _monthly_xticks(dates: list) -> tuple[list, list]:
    """Extract monthly tick positions and labels from date list."""
    seen = set()
    ticks, labels = [], []
    for i, d in enumerate(dates):
        m = str(d)[:6]
        if m not in seen:
            seen.add(m)
            ticks.append(i)
            labels.append(str(d))
    return ticks, labels


def plot_cumulative_return(
    model_ret: pd.Series,
    bench_ret: pd.Series,
    model_label: str = "Tangle Ensemble",
    bench_label: str = "Bench_Equal",
    ax: plt.Axes | None = None,
):
    """Plot cumulative return comparison."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    dates = model_ret.index.tolist()
    ticks, labels = _monthly_xticks(dates)

    ax.plot(dates, model_ret.cumsum(), label=model_label, color="blue", linewidth=1.8)
    ax.plot(dates, bench_ret.cumsum(), label=bench_label, color="orange", linewidth=1.5)
    ax.set_title("Cumulative Return", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()


def plot_relative_improvement(
    model_ret: pd.Series,
    bench_ret: pd.Series,
    ax: plt.Axes | None = None,
):
    """Plot relative improvement (model - bench) cumulative."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    diff = model_ret - bench_ret
    dates = diff.index.tolist()
    ticks, labels = _monthly_xticks(dates)
    cum_diff = diff.cumsum()

    ax.fill_between(
        range(len(cum_diff)), cum_diff.values, 0,
        where=cum_diff.values >= 0, color="#2a9d8f", alpha=0.35, interpolate=True,
    )
    ax.fill_between(
        range(len(cum_diff)), cum_diff.values, 0,
        where=cum_diff.values < 0, color="#e63946", alpha=0.35, interpolate=True,
    )
    ax.plot(cum_diff.values, color="#264653", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    mean_daily = diff.mean()
    win_rate = (diff > 0).mean() * 100
    ax.set_title(
        f"Relative Improvement (Tangle - Bench)  |  "
        f"Mean={mean_daily:.4f}%/day  WinRate={win_rate:.1f}%",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Cumulative Improvement (%)")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.grid(alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()


# ── 2. IC Time Series ───────────────────────────────────────────────────


def plot_ic_timeseries(
    model_ic: pd.Series,
    bench_ic: pd.Series,
    window: int = MA_WINDOW,
    ax: plt.Axes | None = None,
):
    """Plot IC rolling average comparison."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    if window > 1:
        model_ma = model_ic.rolling(window, min_periods=1).mean()
        bench_ma = bench_ic.rolling(window, min_periods=1).mean()
    else:
        model_ma = model_ic
        bench_ma = bench_ic

    dates = model_ic.index.tolist()
    ticks, labels = _monthly_xticks(dates)

    ax.plot(model_ma.values, label="Tangle Ensemble", color="#e63946", linewidth=1.5)
    ax.plot(bench_ma.values, label="Bench_Equal", color="#457b9d", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(
        f"Daily IC Comparison ({window}-day MA)" if window > 1 else "Daily IC Comparison",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("IC")
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()


# ── 3. Summary Bar Chart ────────────────────────────────────────────────


def plot_summary_bars(metrics_dict: dict[str, dict], ax: plt.Axes | None = None):
    """Bar chart comparing IC, ICIR, top_return across methods."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    names = list(metrics_dict.keys())
    x = np.arange(len(names))
    metric_keys = ["IC", "ICIR", "top_return", "top_return_stability"]
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
    w = 0.8 / len(metric_keys)

    for j, mk in enumerate(metric_keys):
        vals = [metrics_dict[n].get(mk, 0) for n in names]
        bars = ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=mk, color=colors[j], alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()


# ── 4. Tangle Tree Drawing (diagnostic) ─────────────────────────────────


def _collect_tree_nodes(node, depth=0):
    """Recursively collect tree nodes with depth info."""
    info = {
        "node": node,
        "depth": depth,
        "is_leaf": node.is_leaf(),
        "label": str(node),
        "p": node.p,
    }
    result = [info]
    if node.left_child is not None:
        result.extend(_collect_tree_nodes(node.left_child, depth + 1))
    if node.right_child is not None:
        result.extend(_collect_tree_nodes(node.right_child, depth + 1))
    return result


def _assign_positions(node, x_range=(0, 1), depth=0, positions=None):
    """Assign (x, y) layout positions to tree nodes."""
    if positions is None:
        positions = {}

    x_mid = (x_range[0] + x_range[1]) / 2
    positions[id(node)] = (x_mid, -depth)

    if node.left_child is not None and node.right_child is not None:
        _assign_positions(node.left_child, (x_range[0], x_mid), depth + 1, positions)
        _assign_positions(node.right_child, (x_mid, x_range[1]), depth + 1, positions)
    elif node.left_child is not None:
        _assign_positions(node.left_child, x_range, depth + 1, positions)
    elif node.right_child is not None:
        _assign_positions(node.right_child, x_range, depth + 1, positions)

    return positions


def plot_tangle_tree(contracted_tree, cluster_labels, cut_names, ax):
    """Draw the contracted tangle tree with cluster sizes on leaves."""
    positions = _assign_positions(contracted_tree.root)
    nodes = _collect_tree_nodes(contracted_tree.root)

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    leaf_idx = 0
    leaf_sizes = {}
    for info in nodes:
        if info["is_leaf"]:
            leaf_sizes[id(info["node"])] = counts[leaf_idx] if leaf_idx < len(counts) else 0
            leaf_idx += 1

    cmap = plt.cm.get_cmap("tab20")

    # Edges
    for info in nodes:
        node = info["node"]
        for child in [node.left_child, node.right_child]:
            if child is not None:
                px, py = positions[id(node)]
                cx, cy = positions[id(child)]
                ax.plot([px, cx], [py, cy], "k-", linewidth=1.0, alpha=0.6)

    # Nodes
    leaf_color_idx = 0
    for info in nodes:
        node = info["node"]
        x, y = positions[id(node)]

        if info["is_leaf"]:
            size = leaf_sizes.get(id(node), 0)
            color = cmap(leaf_color_idx / max(len(unique_labels), 1))
            leaf_color_idx += 1
            node_size = max(200, min(size * 3, 1500))
            ax.scatter(x, y, s=node_size, c=[color], edgecolors="black",
                       linewidth=1.2, zorder=5)
            ax.annotate(f"n={size}", (x, y), textcoords="offset points",
                        xytext=(0, -18), ha="center", fontsize=7, fontweight="bold")
        else:
            ax.scatter(x, y, s=150, c="white", edgecolors="gray",
                       linewidth=1.0, zorder=5)
            if node.last_cut_added_id is not None and node.last_cut_added_id >= 0:
                try:
                    name = cut_names[node.last_cut_added_id]
                except (IndexError, TypeError):
                    name = f"cut_{node.last_cut_added_id}"
                ax.annotate(name, (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=5.5,
                            color="gray", style="italic")

    ax.set_title("Contracted Tangle Tree", fontsize=13, fontweight="bold")
    ax.set_ylabel("Depth (cost order)")
    ax.set_xticks([])
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)


# ── 5. Cluster Weight Heatmap (diagnostic) ──────────────────────────────


def plot_cluster_weights_heatmap(weights, bench_cols, cluster_sizes, ax):
    """Heatmap: rows = clusters, columns = bench models, values = weight."""
    cluster_ids = sorted(weights.keys())
    n_clusters = len(cluster_ids)
    n_benches = len(bench_cols)

    mat = np.zeros((n_clusters, n_benches))
    y_labels = []
    for i, cid in enumerate(cluster_ids):
        mat[i] = weights[cid]
        y_labels.append(f"Cluster {cid}\n(n={cluster_sizes.get(cid, 0)})")

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks(range(n_benches))
    ax.set_xticklabels(bench_cols, fontsize=9)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title("Per-Cluster Model Weights", fontsize=13, fontweight="bold")

    for i in range(n_clusters):
        for j in range(n_benches):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if mat[i, j] > 0.3 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Weight")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Loading data...")
    df = load_data()
    dates = sorted(df["date"].unique().to_list())

    # Load wide data
    ret_data = load_wide_pandas(LABEL_PATH)
    liquid_data = load_wide_pandas(LIQUID_PATH)
    bench_wide = {}
    for name, path in BENCH_PATHS.items():
        bench_wide[name] = pd.read_feather(path).set_index("date")
        bench_wide[name].index = bench_wide[name].index.astype(str)

    # Bench baseline (equal-weight)
    bench_all = ensemble_scores(*bench_wide.values()).mean(axis=1).unstack()

    # ━━ Part A: Tangle tree on a sample date ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sample_date = "20240601"
    avail = [d for d in dates if d >= sample_date]
    sample_date = avail[0] if avail else dates[-1]
    print(f"Sample date for tree visualization: {sample_date}")

    date_df = df.filter(pl.col("date") == sample_date)
    scores = date_df.select(BENCH_COLS).to_numpy().astype(np.float64)
    codes = date_df["code"].to_numpy()

    result = run_tangle_full(scores)
    if result is None:
        sample_date = dates[len(dates) // 2]
        date_df = df.filter(pl.col("date") == sample_date)
        scores = date_df.select(BENCH_COLS).to_numpy().astype(np.float64)
        codes = date_df["code"].to_numpy()
        result = run_tangle_full(scores)

    if result is not None:
        labels = result["labels"]
        contracted = result["contracted_tree"]
        bipartitions = result["bipartitions"]
        cut_names = list(bipartitions.names) if bipartitions.names is not None else []
        unique_labels, counts = np.unique(labels, return_counts=True)

        idx = dates.index(sample_date)
        hist_df = df.filter(pl.col("date").is_in(dates[max(0, idx - LOOKBACK_DAYS):idx]))
        weights = compute_cluster_weights(hist_df, labels, codes, BENCH_COLS)
        cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))

        # Figure 1: Tree + cluster heatmap
        fig1, (ax_tree, ax_hw) = plt.subplots(1, 2, figsize=(20, 8),
                                               gridspec_kw={"width_ratios": [1.5, 1]})
        plot_tangle_tree(contracted, labels, cut_names, ax_tree)
        ax_tree.set_title(
            f"Tangle Tree ({sample_date}, {len(scores)} stocks, "
            f"{len(unique_labels)} clusters)", fontsize=12, fontweight="bold",
        )
        plot_cluster_weights_heatmap(weights, BENCH_COLS, cluster_sizes, ax_hw)
        fig1.tight_layout()
        fig1.savefig(OUTPUT_DIR / "01_tangle_tree_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print("  Saved 01_tangle_tree_weights.png")
    else:
        print("WARNING: Tangle failed on sample date, skipping tree plot.")

    # ━━ Part B: Full backtest evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    eval_dates = [d for d in dates if EVAL_START <= d <= EVAL_END]
    print(f"\nRunning full backtest: {len(eval_dates)} dates...")

    tangle_wide = compute_tangle_scores_wide(df, dates, eval_dates, LOOKBACK_DAYS)

    # Get daily returns and IC for both
    tangle_ret, tangle_ic = get_ret_ic(
        tangle_wide, ret_data, liquid_data, print_quarterly=False,
    )
    bench_ret, bench_ic = get_ret_ic(
        bench_all, ret_data, liquid_data, print_quarterly=False,
    )

    # Align on common dates
    common_dates = tangle_ret.index.intersection(bench_ret.index)
    tangle_ret = tangle_ret.loc[common_dates]
    bench_ret = bench_ret.loc[common_dates]
    tangle_ic = tangle_ic.loc[common_dates]
    bench_ic = bench_ic.loc[common_dates]

    # Convert to percentage for cumsum (÷100)
    tangle_ret_dec = tangle_ret / 100
    bench_ret_dec = bench_ret / 100

    # ── Figure 2: Cumulative return + relative improvement ──
    fig2, (ax_cum, ax_imp) = plt.subplots(2, 1, figsize=(16, 10))
    plot_cumulative_return(tangle_ret_dec, bench_ret_dec, ax=ax_cum)
    plot_relative_improvement(tangle_ret_dec, bench_ret_dec, ax=ax_imp)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "02_cumulative_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved 02_cumulative_return.png")

    # ── Figure 3: IC time series ──
    fig3, ax_ic = plt.subplots(figsize=(16, 6))
    plot_ic_timeseries(tangle_ic, bench_ic, window=20, ax=ax_ic)
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "03_ic_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved 03_ic_timeseries.png")

    # ── Figure 4: Summary bars ──
    tangle_metrics = get_metrics(tangle_wide, ret_data, liquid_data, print_quarterly=False)
    bench_metrics = get_metrics(bench_all, ret_data, liquid_data, print_quarterly=False)

    all_metrics = {"Tangle Ensemble": tangle_metrics, "Bench_Equal": bench_metrics}
    for name, score_df in bench_wide.items():
        all_metrics[name] = get_metrics(
            score_df, ret_data, liquid_data, print_quarterly=False,
        )

    fig4, ax_bar = plt.subplots(figsize=(14, 6))
    plot_summary_bars(all_metrics, ax=ax_bar)
    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / "04_summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("  Saved 04_summary_comparison.png")

    # ── Print summary table ──
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'IC':>8} {'ICIR':>8} {'TopRet%':>10} {'Stability':>10}")
    print("-" * 80)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['IC']:>8.4f} {m['ICIR']:>8.4f} "
              f"{m['top_return']:>10.4f} {m['top_return_stability']:>10.4f}")
    print("=" * 80)
    print(f"\nAll plots saved to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
