"""
Taxonomic annotation utilities using BLAST+ against a local database.

This layer takes cluster assignments and representative sequences
and returns:
    - best BLAST hit per cluster (if any)
    - putative taxonomy / description
    - percent identity / alignment stats
    - a simple "novelty score" combining sequence identity and
      embedding-based cluster novelty.

BLAST is optional: if BLAST_DB is not configured or blastn is not
installed, calls will return empty annotations gracefully.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import BLAST_DB, BLAST_MAX_HITS, BLAST_MIN_IDENTITY


@dataclass
class BlastHit:
    qseqid: str
    sseqid: str
    pident: float
    length: int
    evalue: float
    bitscore: float
    description: str


@dataclass
class ClusterAnnotation:
    cluster_id: int
    size: int
    best_identity: float
    best_description: str
    novelty_score: float  # 0=known, 1=highly novel


def _blast_available() -> bool:
    """Return True if blastn is on PATH and BLAST_DB is configured."""
    if not BLAST_DB:
        return False
    return shutil.which("blastn") is not None


def _write_fasta(seqs: Dict[str, str], path: Path) -> None:
    """Write a small FASTA file of representative sequences."""
    lines = []
    for sid, seq in seqs.items():
        lines.append(f">{sid}")
        lines.append(seq)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_blast_for_representatives(
    reps: Dict[str, str],
    tmp_dir: Path,
) -> Dict[str, BlastHit]:
    """
    Run blastn for a dictionary of representative sequences.

    Returns a dict mapping query ID -> best BlastHit (or omits if no hit).
    """
    results: Dict[str, BlastHit] = {}

    if not _blast_available():
        return results

    tmp_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = tmp_dir / "cluster_reps.fasta"
    out_path = tmp_dir / "cluster_reps_blast.tsv"

    _write_fasta(reps, fasta_path)

    # We request a simple tabular format; stitle gives us some taxonomic text.
    outfmt = "6 qseqid sseqid pident length evalue bitscore stitle"
    cmd = [
        "blastn",
        "-query",
        str(fasta_path),
        "-db",
        BLAST_DB,
        "-outfmt",
        outfmt,
        "-max_target_seqs",
        str(BLAST_MAX_HITS),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=out_path.open("w", encoding="utf-8"))
    except Exception as exc:
        print(f"[taxonomy] BLAST call failed: {exc}")
        return results

    # Parse one best hit per query (top line)
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        qseqid, sseqid, pident, length, evalue, bitscore, stitle = parts[:7]
        pident_f = float(pident)
        if pident_f < BLAST_MIN_IDENTITY:
            continue
        if qseqid in results:
            # already have a hit; keep the first (best-ranked) one
            continue
        results[qseqid] = BlastHit(
            qseqid=qseqid,
            sseqid=sseqid,
            pident=pident_f,
            length=int(length),
            evalue=float(evalue),
            bitscore=float(bitscore),
            description=stitle,
        )

    return results


def choose_cluster_representatives(
    sequences: List[str],
    labels: np.ndarray,
    embedding: np.ndarray,
) -> Dict[int, Tuple[int, str]]:
    """
    For each non-noise cluster, choose the sequence closest to the
    cluster centroid in embedding space as its representative.

    Returns: {cluster_id: (seq_index, sequence_str)}
    """
    labels = np.asarray(labels)
    embedding = np.asarray(embedding, dtype=float)
    reps: Dict[int, Tuple[int, str]] = {}

    for cid in np.unique(labels):
        if cid == -1:
            continue
        mask = labels == cid
        if not np.any(mask):
            continue
        inds = np.where(mask)[0]
        cluster_points = embedding[inds]
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        best_idx_local = int(np.argmin(dists))
        seq_idx = int(inds[best_idx_local])
        reps[int(cid)] = (seq_idx, sequences[seq_idx])

    return reps


def annotate_clusters(
    sequences: List[str],
    labels: np.ndarray,
    embedding: np.ndarray,
    cluster_novelty: Dict[int, float],
    tmp_dir: Path,
) -> Dict[int, ClusterAnnotation]:
    """
    High-level helper to:
        - pick representatives per cluster
        - run BLAST (if configured)
        - compute a novelty score per cluster combining:
            (1 - best_identity/100) and cluster_novelty[cid]
    """
    labels = np.asarray(labels)

    # Pick representatives
    rep_map = choose_cluster_representatives(sequences, labels, embedding)
    rep_seqs = {f"cluster_{cid}": seq for cid, (_, seq) in rep_map.items()}

    blast_hits = run_blast_for_representatives(rep_seqs, tmp_dir=tmp_dir)

    annotations: Dict[int, ClusterAnnotation] = {}
    for cid, (idx, seq) in rep_map.items():
        qid = f"cluster_{cid}"
        size = int((labels == cid).sum())

        if qid in blast_hits:
            hit = blast_hits[qid]
            best_identity = hit.pident
            best_desc = hit.description
        else:
            best_identity = 0.0
            best_desc = "No confident BLAST hit"

        # Novelty score: 0 (not novel) to 1 (very novel)
        # Combine: low sequence identity and high embedding novelty.
        ident_component = 1.0 - (best_identity / 100.0)
        embed_component = float(cluster_novelty.get(cid, 0.0))
        novelty_score = float(0.5 * ident_component + 0.5 * embed_component)

        annotations[cid] = ClusterAnnotation(
            cluster_id=cid,
            size=size,
            best_identity=best_identity,
            best_description=best_desc,
            novelty_score=novelty_score,
        )

    return annotations

