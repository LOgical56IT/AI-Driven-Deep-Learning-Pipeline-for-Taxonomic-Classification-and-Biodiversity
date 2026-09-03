"""
Benchmark the DeepSea eDNA AI pipeline on a given FASTA file.

This measures wall-clock time for:
    - loading sequences
    - k-mer encoding
    - optional deep autoencoder embedding
    - UMAP + HDBSCAN clustering
    - biodiversity metrics + novelty + taxonomy

Results are printed to stdout and optionally written as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biodiversity import summarise_biodiversity
from clustering import cluster_sequences
from config import BLAST_DB
from data_loader import load_fasta
from kmer_encoder import encode_sequences
from models import embed_with_autoencoder
from novelty import sequence_novelty_scores, cluster_novelty_summary
from taxonomy import annotate_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DeepSea eDNA AI pipeline.")
    parser.add_argument("fasta", type=str, help="Input FASTA file.")
    parser.add_argument(
        "--use-deep-model",
        action="store_true",
        help="Use trained autoencoder embeddings if available.",
    )
    parser.add_argument(
        "--ae-path",
        type=str,
        default="models/kmer_ae.pt",
        help="Path to trained autoencoder weights.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write benchmark summary JSON.",
    )
    args = parser.parse_args()

    timings = {}
    t0 = time.perf_counter()
    sequences = load_fasta(args.fasta)
    timings["load_fasta_s"] = time.perf_counter() - t0

    if not sequences:
        raise SystemExit("No sequences found in FASTA.")

    t1 = time.perf_counter()
    X = encode_sequences(sequences)
    timings["encode_kmers_s"] = time.perf_counter() - t1

    # Deep model (if requested and weights exist)
    ae_used = False
    embedding_input = X
    if args.use_deep_model:
        ae_path = Path(args.ae_path)
        if ae_path.exists():
            t2 = time.perf_counter()
            embedding_input = embed_with_autoencoder(X, ae_path)
            timings["autoencoder_embed_s"] = time.perf_counter() - t2
            ae_used = True

    # Clustering
    t3 = time.perf_counter()
    labels, embedding = cluster_sequences(embedding_input)
    timings["cluster_umap_hdbscan_s"] = time.perf_counter() - t3

    # Biodiversity + novelty
    t4 = time.perf_counter()
    biodiversity = summarise_biodiversity(labels)
    novelty_scores = sequence_novelty_scores(embedding, labels)
    cluster_novelty = cluster_novelty_summary(embedding, labels, scores=novelty_scores)
    timings["biodiversity_novelty_s"] = time.perf_counter() - t4

    # Taxonomy (BLAST) if configured
    taxonomy = {}
    if BLAST_DB:
        from taxonomy import annotate_clusters  # local import ensures BLAST config is loaded

        t5 = time.perf_counter()
        taxonomy = annotate_clusters(
            sequences=sequences,
            labels=np.array(labels),
            embedding=np.array(embedding),
            cluster_novelty=cluster_novelty,
            tmp_dir=Path("taxonomy_benchmark_tmp"),
        )
        timings["taxonomy_blast_s"] = time.perf_counter() - t5

    total_time = sum(timings.values())

    summary = {
        "n_sequences": len(sequences),
        "feature_dim": int(X.shape[1]),
        "used_deep_model": ae_used,
        "timings_s": timings,
        "total_time_s": total_time,
        "biodiversity": biodiversity,
        "num_clusters": int(biodiversity["richness"]),
        "noise_count": int((np.array(labels) == -1).sum()),
        "blast_db_configured": bool(BLAST_DB),
        "num_annotated_clusters": len(taxonomy),
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

