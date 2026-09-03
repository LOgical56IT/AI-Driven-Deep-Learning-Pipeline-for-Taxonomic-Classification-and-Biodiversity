"""
Basic biodiversity metrics on clustered eDNA data.

Inputs are assumed to be cluster labels from an unsupervised
algorithm like HDBSCAN (with -1 reserved for "noise"/unclustered).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Tuple

import math
import numpy as np


def _counts_from_labels(labels: Iterable[int], ignore_noise: bool = True) -> Counter:
    """Convert cluster labels to counts, optionally dropping noise (-1)."""
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    if ignore_noise:
        labels = [l for l in labels if l != -1]

    return Counter(labels)


def richness(labels: Iterable[int], ignore_noise: bool = True) -> int:
    """Number of non-noise clusters (taxa richness proxy)."""
    counts = _counts_from_labels(labels, ignore_noise=ignore_noise)
    return len(counts)


def relative_abundance(labels: Iterable[int], ignore_noise: bool = True) -> Dict[int, float]:
    """Relative abundance of each cluster (p_i)."""
    counts = _counts_from_labels(labels, ignore_noise=ignore_noise)
    total = sum(counts.values()) or 1
    return {cid: c / total for cid, c in counts.items()}


def shannon_index(labels: Iterable[int], ignore_noise: bool = True) -> float:
    """Shannon diversity index H' = -sum(p_i * ln p_i)."""
    p = relative_abundance(labels, ignore_noise=ignore_noise)
    return -sum(pi * math.log(pi) for pi in p.values() if pi > 0)


def simpson_index(labels: Iterable[int], ignore_noise: bool = True) -> float:
    """
    Simpson index D = sum(p_i^2) (dominance).
    Often diversity reported as 1 - D.
    """
    p = relative_abundance(labels, ignore_noise=ignore_noise)
    return sum(pi * pi for pi in p.values())


def simpson_diversity(labels: Iterable[int], ignore_noise: bool = True) -> float:
    """Simpson diversity = 1 - D."""
    return 1.0 - simpson_index(labels, ignore_noise=ignore_noise)


def summarise_biodiversity(labels: Iterable[int], ignore_noise: bool = True) -> Dict[str, float]:
    """Convenience helper returning all key metrics in one dict."""
    return {
        "richness": float(richness(labels, ignore_noise=ignore_noise)),
        "shannon": float(shannon_index(labels, ignore_noise=ignore_noise)),
        "simpson_index": float(simpson_index(labels, ignore_noise=ignore_noise)),
        "simpson_diversity": float(simpson_diversity(labels, ignore_noise=ignore_noise)),
    }

