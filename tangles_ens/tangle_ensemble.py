"""Tangle-based ensemble: cluster stocks by model agreement, apply per-cluster weights."""

import sys
import numpy as np
import polars as pl
from functools import partial
from scipy.stats import spearmanr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tangles_src"))

from tangles.data_types import Cuts
from tangles.tree_tangles import (
    tangle_computation,
    ContractedTangleTree,
    compute_soft_predictions_children,
)
from tangles.utils import normalize, compute_hard_predictions
from tangles.cost_functions import gauss_kernel_distance

from config import AGREEMENT_RATIO, N_QUANTILES, PRUNE_DEPTH, MIN_CLUSTER_SIZE


def generate_cuts_from_scores(
    scores: np.ndarray, n_quantiles: int = N_QUANTILES
) -> tuple[np.ndarray, list[str]]:
    """Generate binary cuts from bench scores using quantile thresholds.

    Args:
        scores: (n_stocks, n_benches) array, already rank-normalized to [0, 1].
        n_quantiles: number of quantile splits per bench.

    Returns:
        cuts: (n_cuts, n_stocks) boolean array
        names: list of cut descriptions
    """
    n_stocks, n_benches = scores.shape
    thresholds = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # e.g., [0.2, 0.4, 0.6, 0.8]

    cuts = []
    names = []
    for b in range(n_benches):
        col = scores[:, b]
        for t in thresholds:
            cut = col > t
            # Skip degenerate cuts (all True or all False)
            if cut.sum() < 5 or (~cut).sum() < 5:
                continue
            cuts.append(cut)
            names.append(f"bench{b+1}>{t:.2f}")

    return np.array(cuts, dtype=bool), names


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per column, NaN-safe."""
    result = np.empty_like(scores)
    for j in range(scores.shape[1]):
        col = scores[:, j]
        mask = ~np.isnan(col)
        ranks = np.empty(len(col))
        ranks[:] = np.nan
        from scipy.stats import rankdata
        ranks[mask] = rankdata(col[mask]) / mask.sum()
        result[:, j] = ranks
    return result


def cut_cost_balance(n_samples, cut):
    """Simple cost: penalize imbalanced cuts. Balanced cut = low cost."""
    s = cut.sum()
    ratio = min(s, len(cut) - s) / len(cut)
    return 1.0 - ratio  # 0 = perfectly balanced, 0.5 = maximally imbalanced


def run_tangle_clustering(scores: np.ndarray) -> np.ndarray | None:
    """Run tangle computation on (n_stocks, n_benches) scores.

    Returns:
        cluster_labels: (n_stocks,) integer array, or None if tangle fails.
    """
    n_stocks = scores.shape[0]

    # Step 1: rank normalize
    ranked = rank_normalize(scores)

    # Step 2: generate cuts
    cut_values, cut_names = generate_cuts_from_scores(ranked, n_quantiles=N_QUANTILES)
    if len(cut_values) < 3:
        return None

    # Step 3: wrap as Cuts object (names must be np.array for indexing in _order_cuts)
    bipartitions = Cuts(
        values=cut_values,
        names=np.array(cut_names),
    )

    # Step 4: compute costs and sort — use gaussian kernel distance on ranked features
    cost_fn = partial(gauss_kernel_distance, ranked, None)
    bipartitions.compute_cost_and_order_cuts(cost_fn, verbose=False)

    # Step 5: tangle computation
    agreement = max(int(n_stocks * AGREEMENT_RATIO), 10)
    try:
        tree = tangle_computation(
            cuts=bipartitions,
            agreement=agreement,
            verbose=0,
        )
    except Exception:
        return None

    if tree is None or not tree.maximals:
        return None

    # Step 6: contract and prune
    try:
        contracted = ContractedTangleTree(tree)
        contracted.prune(PRUNE_DEPTH)
        contracted.calculate_setP()
    except Exception:
        return None

    # Step 7: soft predictions → hard predictions
    weight = np.exp(-normalize(bipartitions.costs))
    compute_soft_predictions_children(
        node=contracted.root,
        cuts=bipartitions,
        weight=weight,
    )

    labels, _ = compute_hard_predictions(contracted, cuts=bipartitions)
    return labels


def run_tangle_full(scores: np.ndarray) -> dict | None:
    """Run tangle computation and return all intermediate objects for visualization.

    Returns dict with keys: labels, contracted_tree, bipartitions, weight, ranked
    """
    n_stocks = scores.shape[0]
    ranked = rank_normalize(scores)

    cut_values, cut_names = generate_cuts_from_scores(ranked, n_quantiles=N_QUANTILES)
    if len(cut_values) < 3:
        return None

    bipartitions = Cuts(values=cut_values, names=np.array(cut_names))
    cost_fn = partial(gauss_kernel_distance, ranked, None)
    bipartitions.compute_cost_and_order_cuts(cost_fn, verbose=False)

    agreement = max(int(n_stocks * AGREEMENT_RATIO), 10)
    try:
        tree = tangle_computation(cuts=bipartitions, agreement=agreement, verbose=0)
    except Exception:
        return None

    if tree is None or not tree.maximals:
        return None

    try:
        contracted = ContractedTangleTree(tree)
        contracted.prune(PRUNE_DEPTH, verbose=False)
        contracted.calculate_setP()
    except Exception:
        return None

    weight = np.exp(-normalize(bipartitions.costs))
    compute_soft_predictions_children(node=contracted.root, cuts=bipartitions, weight=weight)
    labels, _ = compute_hard_predictions(contracted, cuts=bipartitions)

    return {
        "labels": labels,
        "contracted_tree": contracted,
        "bipartitions": bipartitions,
        "weight": weight,
        "ranked": ranked,
        "agreement": agreement,
    }


def compute_cluster_weights(
    hist_df: pl.DataFrame,
    cluster_labels: np.ndarray,
    current_codes: np.ndarray,
    bench_cols: list[str],
) -> dict[int, np.ndarray]:
    """Compute per-cluster model weights using historical IC.

    For each cluster, look at the stocks in that cluster, compute
    the historical Spearman correlation of each bench with labels,
    and use max(IC, 0) as weight (normalized to sum=1).

    Args:
        hist_df: historical data (date, code, bench1..bench6, label)
        cluster_labels: (n_stocks,) cluster assignment for current date
        current_codes: (n_stocks,) stock codes matching cluster_labels
        bench_cols: list of bench column names

    Returns:
        dict mapping cluster_id → weight array (n_benches,)
    """
    n_benches = len(bench_cols)
    equal_weights = np.ones(n_benches) / n_benches

    code_to_cluster = dict(zip(current_codes, cluster_labels))
    unique_clusters = np.unique(cluster_labels)

    weights = {}
    for c in unique_clusters:
        cluster_codes = [code for code, cl in code_to_cluster.items() if cl == c]
        if len(cluster_codes) < MIN_CLUSTER_SIZE:
            weights[c] = equal_weights
            continue

        # Filter historical data to stocks in this cluster
        cluster_hist = hist_df.filter(pl.col("code").is_in(cluster_codes))
        if cluster_hist.height < MIN_CLUSTER_SIZE * 5:
            weights[c] = equal_weights
            continue

        # Compute IC (Spearman) per bench
        ics = np.zeros(n_benches)
        label_arr = cluster_hist["label"].to_numpy()
        for i, bcol in enumerate(bench_cols):
            bench_arr = cluster_hist[bcol].to_numpy().astype(np.float64)
            valid = ~(np.isnan(bench_arr) | np.isnan(label_arr))
            if valid.sum() > 30:
                corr, _ = spearmanr(bench_arr[valid], label_arr[valid])
                ics[i] = corr if not np.isnan(corr) else 0.0

        # Weight = max(IC, 0), fallback to equal if all zero
        w = np.maximum(ics, 0.0)
        if w.sum() < 1e-10:
            weights[c] = equal_weights
        else:
            weights[c] = w / w.sum()

    return weights


def ensemble_one_date(
    date_df: pl.DataFrame,
    hist_df: pl.DataFrame,
    bench_cols: list[str],
) -> pl.DataFrame:
    """Run tangle ensemble for a single cross-section date.

    Args:
        date_df: single-date data (code, bench1..bench6, label)
        hist_df: historical lookback data for weight calibration
        bench_cols: bench column names

    Returns:
        DataFrame with columns (code, ensemble_score, label)
    """
    codes = date_df["code"].to_numpy()
    scores = date_df.select(bench_cols).to_numpy().astype(np.float64)

    # Run tangle clustering
    labels = run_tangle_clustering(scores)

    if labels is None:
        # Fallback: equal-weight average
        ensemble_score = np.nanmean(scores, axis=1)
    else:
        # Compute per-cluster weights from history
        weights = compute_cluster_weights(hist_df, labels, codes, bench_cols)

        # Apply cluster-specific weights
        ensemble_score = np.zeros(len(codes))
        for i, (code, cluster) in enumerate(zip(codes, labels)):
            w = weights.get(cluster, np.ones(len(bench_cols)) / len(bench_cols))
            row = scores[i]
            # Replace NaN with 0 for weighted sum
            row_clean = np.where(np.isnan(row), 0.0, row)
            ensemble_score[i] = np.dot(w, row_clean)

    return pl.DataFrame({
        "code": codes,
        "ensemble_score": ensemble_score,
        "label": date_df["label"].to_numpy(),
    })
