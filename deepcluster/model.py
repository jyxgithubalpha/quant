"""DeepCluster V2 model for multi-factor financial data.

Architecture:
    Encoder MLP:     input_dim → [512, 256, 128] with BN + ReLU + Dropout
    Projection Head: 128 → 64 (L2-normalized, for clustering)
    Prediction Head: 128 → 1 (return prediction)

Multi-clustering: K-means with K ∈ {50, 100, 200} on projected embeddings.
Cluster logits via cosine similarity to centroids / temperature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):
    """Factor encoder: input_dim → embedding_dim with BatchNorm + ReLU."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """MLP projection head: embedding → projection space (for clustering)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, dim=1)  # L2 normalize for cosine similarity


class DeepClusterV2(nn.Module):
    """DeepCluster V2 model with multi-clustering + return prediction.

    Training loop (external):
        1. Forward all data → projected embeddings z
        2. K-means on z for each K → centroids + assignments
        3. Set centroids via set_centroids()
        4. Mini-batch: forward → cluster logits + pred → loss → backprop

    Loss = cluster_weight * avg_CE(logits, pseudo_labels) + predict_weight * MSE(pred, label)
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        projection_dim: int,
        cluster_ks: list[int],
        temperature: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cluster_ks = cluster_ks
        self.temperature = temperature

        self.encoder = EncoderMLP(input_dim, encoder_dims, dropout)
        self.projection = ProjectionHead(self.encoder.output_dim, projection_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Centroid buffers for each K (set externally after K-means)
        for k in cluster_ks:
            self.register_buffer(f"centroids_{k}", torch.zeros(k, projection_dim))

    def set_centroids(self, k: int, centroids: torch.Tensor):
        """Update centroids for a given K after K-means."""
        normed = F.normalize(centroids.float(), dim=1)
        getattr(self, f"centroids_{k}").copy_(normed)

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (embedding, projection, cluster_logits_dict, prediction)."""
        h = self.encoder(x)                          # (B, embed_dim)
        z = self.projection(h)                       # (B, proj_dim), L2-normed
        pred = self.predictor(h).squeeze(-1)         # (B,)

        logits = {}
        for k in self.cluster_ks:
            centroids = getattr(self, f"centroids_{k}")
            logits[k] = z @ centroids.T / self.temperature  # (B, K)

        return h, z, logits, pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding only (for feature extraction)."""
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized projection (for clustering)."""
        h = self.encoder(x)
        return self.projection(h)

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get return prediction only (for inference)."""
        h = self.encoder(x)
        return self.predictor(h).squeeze(-1)
