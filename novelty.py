"""
Novelty / out-of-distribution (OOD) scoring utilities.

These functions operate on a 2D embedding (UMAP or deep latent space)
and cluster labels (e.g. from HDBSCAN) to estimate:
    - per-sequence novelty score
    - per-cluster average novelty

The intuition:
    - sequences far from their cluster centroid are more "novel"
    - sequences marked as noise (-1) are highly novel
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def compute_cluster_centroids(embedding: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Return centroid per cluster ID (excluding noise label -1)."""
    centroids: Dict[int, np.ndarray] = {}
    for cid in np.unique(labels):
        if cid == -1:
            continue
        mask = labels == cid
        if not np.any(mask):
            continue
        centroids[int(cid)] = embedding[mask].mean(axis=0)
    return centroids


def sequence_novelty_scores(
    embedding: np.ndarray,
    labels: np.ndarray,
    noise_boost: float = 3.0,
) -> np.ndarray:
    """
    Compute a novelty score per sequence.

    - For clustered sequences: Euclidean distance to their cluster centroid.
    - For noise sequences (label -1): distance to global mean * noise_boost.
    - Scores are then normalised to [0, 1].
    """
    embedding = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels)
    n, d = embedding.shape

    centroids = compute_cluster_centroids(embedding, labels)
    global_center = embedding.mean(axis=0)

    scores = np.zeros(n, dtype=float)
    for i in range(n):
        cid = int(labels[i])
        x = embedding[i]
        if cid == -1 or cid not in centroids:
            # noise or orphan point: distance from global centre, boosted
            dist = np.linalg.norm(x - global_center) * noise_boost
        else:
            dist = np.linalg.norm(x - centroids[cid])
        scores[i] = dist

    # normalise to [0, 1] for interpretability
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


def cluster_novelty_summary(
    embedding: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
) -> Dict[int, float]:
    """
    Average novelty score per cluster ID (excluding noise).
    If scores is None, it is recomputed.
    """
    labels = np.asarray(labels)
    if scores is None:
        scores = sequence_novelty_scores(embedding, labels)
    scores = np.asarray(scores)

    result: Dict[int, float] = {}
    for cid in np.unique(labels):
        if cid == -1:
            continue
        mask = labels == cid
        if not np.any(mask):
            continue
        result[int(cid)] = float(scores[mask].mean())
    return result

