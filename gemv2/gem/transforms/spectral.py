"""
SpectralCluster -- spectral clustering feature augmentation.

Ported from spectral_cluster/clustering.py into the gem transform interface.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseTransform

log = logging.getLogger(__name__)


class SpectralCluster(BaseTransform):
    """
    Spectral clustering feature augmentation.

    mode='onehot':    X -> [X, cluster_one_hot]      (+n_clusters cols)
    mode='embedding': X -> [X, spectral_embedding]    (+n_components cols)

    For large datasets (> subsample_n), spectral decomposition is performed
    on a random subsample, and NearestCentroid propagates cluster assignments
    to the full dataset.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        n_components: int = 10,
        n_neighbors: int = 20,
        mode: str = "onehot",
        subsample_n: int = 5000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.subsample_n = subsample_n
        self.seed = seed
        # fitted state
        self._eigenvectors: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._kmeans = None
        self._centroid_clf = None
        self._embedding_normed: Optional[np.ndarray] = None

    def fit(self, X, y, keys=None):
        from scipy.sparse import diags, eye as speye
        from scipy.sparse.linalg import eigsh
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestCentroid, kneighbors_graph

        rng = np.random.RandomState(self.seed)

        # subsample for scalability
        if X.shape[0] > self.subsample_n:
            idx = rng.choice(X.shape[0], self.subsample_n, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        n = X_sample.shape[0]
        safe_k = max(2, min(self.n_neighbors, n - 1))
        n_eig = min(self.n_components + 1, n - 1)

        # 1. kNN affinity graph (symmetric)
        A = kneighbors_graph(X_sample, n_neighbors=safe_k, mode="connectivity",
                             include_self=False)
        A = 0.5 * (A + A.T)

        # 2. normalized Laplacian  L = I - D^{-1/2} A D^{-1/2}
        degrees = np.maximum(np.array(A.sum(axis=1)).flatten(), 1e-10)
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        L = speye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        # 3. eigen-decomposition (smallest eigenvalues)
        eigenvalues, eigenvectors = eigsh(L, k=n_eig, which="SM",
                                          maxiter=5000, tol=1e-6)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

        # 4. spectral embedding (skip trivial eigenvector at index 0)
        n_embed = min(self.n_clusters, eigenvectors.shape[1] - 1)
        embedding = eigenvectors[:, 1:n_embed + 1]
        norms = np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-10)
        self._embedding_normed = embedding / norms

        # 5. KMeans on spectral embedding
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed,
                              n_init=10, max_iter=300)
        self._kmeans.fit(self._embedding_normed)

        # 6. NearestCentroid for cluster propagation to new data
        self._centroid_clf = NearestCentroid()
        self._centroid_clf.fit(X_sample, self._kmeans.labels_)

        sizes = dict(zip(*np.unique(self._kmeans.labels_, return_counts=True)))
        log.info("SpectralCluster fit: n_sample=%d, n_clusters=%d, sizes=%s",
                 n, self.n_clusters, sizes)
        return self

    def transform(self, X, y, keys=None):
        labels = self._centroid_clf.predict(X)

        if self.mode == "onehot":
            aug = np.zeros((X.shape[0], self.n_clusters), dtype=X.dtype)
            aug[np.arange(X.shape[0]), labels] = 1.0
        else:
            # embedding mode: project to spectral space via cluster centers
            aug = self._kmeans.cluster_centers_[labels]  # (n, n_clusters) in embedding space

        return np.hstack([X, aug]), y

    def get_output_feature_names(self, input_names: List[str]) -> List[str]:
        if self.mode == "onehot":
            new_names = [f"spectral_cluster_{i}" for i in range(self.n_clusters)]
        else:
            n_emb = self._kmeans.cluster_centers_.shape[1] if self._kmeans else self.n_components
            new_names = [f"spectral_emb_{i}" for i in range(n_emb)]
        return input_names + new_names
