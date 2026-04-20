"""
clustering.py — Spectral decomposition, cluster assignment, and feature building.
"""
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags, eye as speye
from sklearn.neighbors import kneighbors_graph, NearestCentroid
from sklearn.cluster import KMeans


def spectral_decompose(X: np.ndarray, n_clusters: int = 8,
                       n_neighbors: int = 10, n_jobs: int = 4,
                       seed: int = 42):
    """
    Manual spectral clustering pipeline:
      1. Build symmetric kNN affinity graph
      2. Compute normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
      3. Solve for the smallest n_clusters+1 eigenvalues/vectors
      4. KMeans on the row-normalized spectral embedding (skip trivial eigenvector)
    Returns: (cluster_labels, eigenvalues, eigenvectors, affinity_matrix)
    """
    n = X.shape[0]
    safe_k = max(2, min(int(n_neighbors), n - 1))
    safe_jobs = max(1, min(int(n_jobs), 8))

    print(f"  Building kNN affinity graph (k={safe_k}) ...")
    A = kneighbors_graph(X, n_neighbors=safe_k, mode="connectivity",
                         include_self=False, n_jobs=safe_jobs)
    A = 0.5 * (A + A.T)  # symmetrize

    # Normalized Laplacian
    degrees = np.maximum(np.array(A.sum(axis=1)).flatten(), 1e-10)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L_sym = speye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    n_eig = min(n_clusters + 1, n - 1)
    print(f"  Eigen-decomposition (computing {n_eig} smallest eigenvalues) ...")
    eigenvalues, eigenvectors = eigsh(L_sym, k=n_eig, which="SM",
                                      maxiter=5000, tol=1e-6)

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Spectral embedding: skip the trivial (constant) eigenvector at index 0
    embedding = eigenvectors[:, 1:n_clusters + 1]
    norms = np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-10)
    embedding_normed = embedding / norms

    # KMeans on the embedding
    print(f"  KMeans on spectral embedding (k={n_clusters}) ...")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(embedding_normed)

    print(f"  Spectral decomposition done, cluster sizes: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels, eigenvalues, eigenvectors, A


def assign_clusters(X_train: np.ndarray, train_labels: np.ndarray,
                    X_new: np.ndarray) -> np.ndarray:
    """Propagate cluster assignments to new data via NearestCentroid."""
    clf = NearestCentroid()
    clf.fit(X_train, train_labels)
    return clf.predict(X_new)


def build_features_with_cluster(X: np.ndarray, cluster_labels: np.ndarray,
                                n_clusters: int) -> np.ndarray:
    """Concatenate raw features with cluster one-hot encoding."""
    onehot = np.zeros((len(cluster_labels), n_clusters), dtype=np.float32)
    onehot[np.arange(len(cluster_labels)), cluster_labels] = 1.0
    return np.concatenate([X, onehot], axis=1)
