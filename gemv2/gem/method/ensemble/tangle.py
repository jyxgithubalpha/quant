"""
TangleEnsemble -- cross-sectional clustering via tangles, per-cluster IC-weighted combination.

Ported from tangles_ens/tangle_ensemble.py into the gem ensemble strategy interface.
"""

import logging
from functools import partial
from typing import Dict, Optional

import numpy as np
from scipy.stats import rankdata, spearmanr

from ...core.data import SplitView
from .strategy import BaseEnsembleStrategy

log = logging.getLogger(__name__)


class TangleEnsemble(BaseEnsembleStrategy):
    """
    Per-date cross-sectional clustering via tangles, then per-cluster
    IC-weighted model combination.

    Requires the ``tangles_src`` package (pip install -e ../tangles_src).
    """

    def __init__(
        self,
        agreement_ratio: float = 0.05,
        n_quantiles: int = 4,
        prune_depth: int = 1,
        min_cluster_size: int = 30,
    ) -> None:
        self.agreement_ratio = agreement_ratio
        self.n_quantiles = n_quantiles
        self.prune_depth = prune_depth
        self.min_cluster_size = min_cluster_size
        # fitted state: per-date cluster weight matrices
        self._cluster_weights_by_date: Dict[int, Optional[np.ndarray]] = {}
        self._fallback_weights: Optional[np.ndarray] = None
        self._model_names: list = []

    def fit(self, predictions, view):
        dates = view.keys["date"].to_numpy()
        y = view.y.ravel()
        self._model_names = list(predictions.keys())
        n_models = len(self._model_names)
        scores = np.column_stack([predictions[n] for n in self._model_names])

        all_weights = []
        for date in np.unique(dates):
            mask = dates == date
            scores_day = scores[mask]
            y_day = y[mask]

            labels = self._cluster_stocks(scores_day)
            if labels is None:
                self._cluster_weights_by_date[int(date)] = None
                continue

            w = self._compute_cluster_weights(labels, scores_day, y_day, n_models)
            self._cluster_weights_by_date[int(date)] = w
            all_weights.append(w)

        if all_weights:
            self._fallback_weights = np.mean(all_weights, axis=0)
        else:
            self._fallback_weights = None

        log.info("TangleEnsemble fit: %d dates, %d with clusters",
                 len(np.unique(dates)), len(all_weights))
        return self

    def combine(self, predictions, view):
        dates = view.keys["date"].to_numpy()
        names = list(predictions.keys())
        scores = np.column_stack([predictions[n] for n in names])
        result = np.zeros(len(dates), dtype=np.float64)

        for date in np.unique(dates):
            mask = dates == date
            scores_day = scores[mask]
            w = self._cluster_weights_by_date.get(int(date), self._fallback_weights)

            if w is None:
                result[mask] = np.mean(scores_day, axis=1)
            else:
                result[mask] = self._apply_cluster_weights(scores_day, w)

        return result

    # -- internal methods -------------------------------------------------

    def _cluster_stocks(self, scores_day: np.ndarray) -> Optional[np.ndarray]:
        """Run tangle clustering on one cross-section.  Returns labels or None."""
        n_stocks = scores_day.shape[0]
        if n_stocks < self.min_cluster_size * 2:
            return None

        try:
            from tangles.data_types import Cuts
            from tangles.tree_tangles import (
                ContractedTangleTree,
                compute_soft_predictions_children,
                tangle_computation,
            )
            from tangles.utils import compute_hard_predictions, normalize
            from tangles.cost_functions import gauss_kernel_distance
        except ImportError:
            raise ImportError(
                "tangles_src is required for TangleEnsemble. "
                "Install with: pip install -e ../tangles_src"
            )

        # rank normalize
        ranked = self._rank_normalize(scores_day)

        # generate binary cuts from quantile thresholds
        cuts_arr, cut_names = self._generate_cuts(ranked)
        if len(cuts_arr) < 3:
            return None

        bipartitions = Cuts(values=cuts_arr, names=np.array(cut_names))
        cost_fn = partial(gauss_kernel_distance, ranked, None)
        bipartitions.compute_cost_and_order_cuts(cost_fn, verbose=False)

        agreement = max(int(n_stocks * self.agreement_ratio), 10)

        try:
            tree = tangle_computation(cuts=bipartitions, agreement=agreement, verbose=0)
        except Exception:
            return None

        if tree is None or not tree.maximals:
            return None

        try:
            contracted = ContractedTangleTree(tree)
            contracted.prune(self.prune_depth)
            contracted.calculate_setP()
        except Exception:
            return None

        weight = np.exp(-normalize(bipartitions.costs))
        compute_soft_predictions_children(node=contracted.root, cuts=bipartitions, weight=weight)
        labels, _ = compute_hard_predictions(contracted, cuts=bipartitions)
        return labels

    def _generate_cuts(self, ranked: np.ndarray):
        """Generate binary cuts from quantile thresholds."""
        n_stocks, n_models = ranked.shape
        thresholds = np.linspace(0, 1, self.n_quantiles + 2)[1:-1]

        cuts = []
        names = []
        for b in range(n_models):
            col = ranked[:, b]
            for t in thresholds:
                cut = col > t
                if cut.sum() < 5 or (~cut).sum() < 5:
                    continue
                cuts.append(cut)
                names.append(f"model{b}>{t:.2f}")

        if not cuts:
            return np.array([], dtype=bool), []
        return np.array(cuts, dtype=bool), names

    @staticmethod
    def _rank_normalize(scores: np.ndarray) -> np.ndarray:
        """Cross-sectional percentile rank per column, NaN-safe."""
        result = np.empty_like(scores, dtype=np.float64)
        for j in range(scores.shape[1]):
            col = scores[:, j]
            valid = ~np.isnan(col)
            ranks = np.full(len(col), np.nan)
            if valid.sum() > 0:
                ranks[valid] = rankdata(col[valid]) / valid.sum()
            result[:, j] = ranks
        return result

    def _compute_cluster_weights(
        self, labels: np.ndarray, scores: np.ndarray, y: np.ndarray, n_models: int,
    ) -> np.ndarray:
        """Per-cluster Spearman IC for each model → weight matrix (n_clusters, n_models)."""
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)
        weights = np.ones((n_clusters, n_models)) / n_models

        for ci, c in enumerate(unique_clusters):
            mask = labels == c
            if mask.sum() < self.min_cluster_size:
                continue
            y_c = y[mask]
            ics = np.zeros(n_models)
            for m in range(n_models):
                s = scores[mask, m]
                valid = ~(np.isnan(s) | np.isnan(y_c))
                if valid.sum() > 10:
                    corr, _ = spearmanr(s[valid], y_c[valid])
                    ics[m] = corr if np.isfinite(corr) else 0.0
            w = np.maximum(ics, 0.0)
            if w.sum() > 1e-10:
                weights[ci] = w / w.sum()

        return weights

    def _apply_cluster_weights(self, scores_day: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply cluster weights to combine model scores.

        Uses a simple heuristic: assign each stock to the nearest cluster
        based on its score profile similarity to cluster centroids.
        For fitted (val) dates, re-cluster; for unseen dates, use global weights.
        """
        # Simple approach: re-cluster via tangle on this day's scores
        labels = self._cluster_stocks(scores_day)
        if labels is None:
            # Fallback: use mean of all cluster weights
            avg_w = weights.mean(axis=0)
            avg_w = avg_w / (avg_w.sum() + 1e-10)
            return scores_day @ avg_w

        unique_clusters = np.unique(labels)
        result = np.zeros(scores_day.shape[0], dtype=np.float64)
        n_weight_clusters = weights.shape[0]

        for ci, c in enumerate(unique_clusters):
            mask = labels == c
            wi = ci if ci < n_weight_clusters else n_weight_clusters - 1
            w = weights[wi]
            scores_clean = np.nan_to_num(scores_day[mask], nan=0.0)
            result[mask] = scores_clean @ w

        return result
