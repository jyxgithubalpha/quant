"""
Spectral embedding propagation for XGBoost ablation.

Provides two feature augmentation modes:
  - one-hot cluster features (wraps spectral_cluster/clustering.py)
  - continuous spectral embedding features (eigenvector coordinates)
"""

import sys
import os
import numpy as np
from sklearn.neighbors import NearestCentroid

# Add spectral_cluster to import path
_SPECTRAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spectral_cluster")
if _SPECTRAL_DIR not in sys.path:
    sys.path.insert(0, _SPECTRAL_DIR)

from clustering import spectral_decompose, assign_clusters, build_features_with_cluster


def augment_with_cluster_onehot(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_clusters: int = 8,
    n_neighbors: int = 10,
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment feature matrices with one-hot spectral cluster labels.

    Steps:
      1. Subsample X_train → spectral_decompose → cluster centroids
      2. Propagate cluster labels to train/val/test via NearestCentroid
      3. Concatenate one-hot encoding with original features

    Returns (X_train_aug, X_val_aug, X_test_aug) with shape (n, d + n_clusters).
    """
    # Subsample for spectral decomposition
    n_train = X_train.shape[0]
    actual_n = min(n_samples, n_train)
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n_train, size=actual_n, replace=False)
    X_sample = X_train[sample_idx]

    # Spectral clustering on sample
    labels_sample, _, _, _ = spectral_decompose(
        X_sample, n_clusters=n_clusters, n_neighbors=n_neighbors, seed=seed
    )

    # Propagate to all data
    train_labels = assign_clusters(X_sample, labels_sample, X_train)
    val_labels = assign_clusters(X_sample, labels_sample, X_val)
    test_labels = assign_clusters(X_sample, labels_sample, X_test)

    # Build augmented features
    X_train_aug = build_features_with_cluster(X_train, train_labels, n_clusters)
    X_val_aug = build_features_with_cluster(X_val, val_labels, n_clusters)
    X_test_aug = build_features_with_cluster(X_test, test_labels, n_clusters)

    print(f"  [spectral_cluster] {X_train.shape[1]} → {X_train_aug.shape[1]} features "
          f"(+{n_clusters} one-hot clusters)")
    return X_train_aug, X_val_aug, X_test_aug


def augment_with_spectral_embedding(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_clusters: int = 8,
    n_neighbors: int = 10,
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment feature matrices with continuous spectral embedding coordinates.

    Unlike one-hot, this preserves topological distance information from the
    spectral decomposition. New data points are projected via NearestCentroid
    mapping in the original feature space, then assigned the centroid's embedding.

    Returns (X_train_aug, X_val_aug, X_test_aug) with shape (n, d + n_clusters).
    """
    n_train = X_train.shape[0]
    actual_n = min(n_samples, n_train)
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n_train, size=actual_n, replace=False)
    X_sample = X_train[sample_idx]

    # Spectral decomposition
    labels_sample, eigenvalues, eigenvectors, _ = spectral_decompose(
        X_sample, n_clusters=n_clusters, n_neighbors=n_neighbors, seed=seed
    )

    # Extract embedding (skip trivial first eigenvector)
    n_eig = min(n_clusters, eigenvectors.shape[1] - 1)
    embedding_sample = eigenvectors[:, 1:n_eig + 1]
    # Row normalize
    norms = np.maximum(np.linalg.norm(embedding_sample, axis=1, keepdims=True), 1e-10)
    embedding_sample = embedding_sample / norms

    # Compute per-cluster centroid embeddings
    unique_labels = np.unique(labels_sample)
    centroid_embeddings = {}
    for lbl in unique_labels:
        mask = labels_sample == lbl
        centroid_embeddings[lbl] = embedding_sample[mask].mean(axis=0)

    # Propagate: assign cluster label via NearestCentroid, then map to centroid embedding
    def _propagate_embedding(X_source, source_labels, X_target):
        clf = NearestCentroid()
        clf.fit(X_source, source_labels)
        target_labels = clf.predict(X_target)
        emb = np.zeros((len(X_target), n_eig), dtype=np.float32)
        for i, lbl in enumerate(target_labels):
            emb[i] = centroid_embeddings.get(lbl, np.zeros(n_eig))
        return emb

    emb_train = _propagate_embedding(X_sample, labels_sample, X_train)
    emb_val = _propagate_embedding(X_sample, labels_sample, X_val)
    emb_test = _propagate_embedding(X_sample, labels_sample, X_test)

    X_train_aug = np.concatenate([X_train, emb_train], axis=1).astype(np.float32)
    X_val_aug = np.concatenate([X_val, emb_val], axis=1).astype(np.float32)
    X_test_aug = np.concatenate([X_test, emb_test], axis=1).astype(np.float32)

    print(f"  [spectral_embedding] {X_train.shape[1]} → {X_train_aug.shape[1]} features "
          f"(+{n_eig} embedding dims)")
    return X_train_aug, X_val_aug, X_test_aug
