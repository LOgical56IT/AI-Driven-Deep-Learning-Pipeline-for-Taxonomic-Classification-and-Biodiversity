from typing import List
import itertools
import numpy as np
from config import KMER_SIZE


def build_kmer_index(k: int) -> dict:
    """Create a dictionary mapping k-mers to indices."""
    kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def encode_sequence(sequence: str, k: int, kmer_index: dict) -> np.ndarray:
    """Convert one DNA sequence into a normalized k-mer frequency vector."""
    sequence = sequence.upper()
    vec = np.zeros(len(kmer_index), dtype=np.float32)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1

    if vec.sum() > 0:
        vec /= vec.sum()

    return vec


def encode_sequences(sequences: List[str], k: int | None = None) -> np.ndarray:
    """Encode multiple sequences into a feature matrix."""
    if k is None:
        k = KMER_SIZE

    kmer_index = build_kmer_index(k)
    return np.array([encode_sequence(seq, k, kmer_index) for seq in sequences])
