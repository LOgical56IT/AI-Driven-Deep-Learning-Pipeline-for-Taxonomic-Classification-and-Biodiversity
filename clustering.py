import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from config import MIN_CLUSTER_SIZE


def cluster_sequences(X: np.ndarray):
    """
    Reduce dimensionality using UMAP and cluster using HDBSCAN.

    Args:
        X: High-dimensional feature matrix (e.g. k-mer frequencies).

    Returns:
        labels: cluster labels for each sequence (-1 = noise)
        embedding: 2D embedding for visualization
    """

    # Handle very small datasets explicitly to avoid UMAP/ HDBSCAN edge cases.
    n_samples = int(X.shape[0])
    if n_samples < 3:
        # With fewer than 3 points, density-based clustering is not meaningful.
        # Return a trivial 2D embedding and unique labels per point.
        embedding = np.zeros((n_samples, 2), dtype=float)
        labels = np.arange(n_samples, dtype=int)
        return labels, embedding

    # Step 1: Dimensionality reduction
    # For tiny demo datasets, UMAP's n_neighbors must be < n_samples.
    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    embedding = reducer.fit_transform(X)

    # Step 2: Density-based clustering
    # HDBSCAN's min_cluster_size / min_samples must be <= n_samples
    effective_min = max(2, min(MIN_CLUSTER_SIZE, n_samples))
    clusterer = HDBSCAN(
        min_cluster_size=effective_min,
        min_samples=effective_min,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding)

    return labels, embedding
