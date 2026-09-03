from __future__ import annotations

from pathlib import Path
from typing import List
import json
import time

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from biodiversity import summarise_biodiversity
from clustering import cluster_sequences
from data_loader import load_fasta
from kmer_encoder import encode_sequences
from models import embed_with_autoencoder
from novelty import sequence_novelty_scores, cluster_novelty_summary
from taxonomy import annotate_clusters


BASE_DIR = Path(__file__).parent
DATA_TMP_DIR = BASE_DIR / "uploaded_data"
DATA_TMP_DIR.mkdir(exist_ok=True)

DEBUG_LOG_PATH = BASE_DIR / ".cursor" / "debug.log"


# region agent log
def _debug_log(message: str, data: dict | None = None, location: str = "web_app.py", hypothesis_id: str = "A", run_id: str = "default") -> None:
    """Append a single NDJSON debug log line to the shared debug file."""
    try:
        payload = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Never let logging break the API
        pass
# endregion agent log

app = FastAPI(
    title="DeepSea eDNA AI",
    description="Upload eDNA reads (FASTA) and explore unsupervised clusters and biodiversity metrics.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve a very simple HTML frontend."""
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found.")
    return html_path.read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze_fasta(
    file: UploadFile = File(...),
    use_deep_model: bool = Form(False),
):
    """
    Upload a FASTA file of eDNA reads, run k-mer encoding + UMAP + HDBSCAN,
    and return cluster labels, 2D embedding, and biodiversity metrics.
    """
    _debug_log(
        "analyze_fasta called",
        data={"filename": file.filename, "use_deep_model": use_deep_model},
        location="web_app.py:analyze_fasta",
        hypothesis_id="H1",
        run_id="pre-fix",
    )

    if not file.filename.lower().endswith((".fa", ".fasta", ".fna")):
        raise HTTPException(status_code=400, detail="Please upload a FASTA file (.fa, .fasta, .fna).")

    dest_path = DATA_TMP_DIR / file.filename
    content = await file.read()
    dest_path.write_bytes(content)

    sequences: List[str] = load_fasta(str(dest_path))
    _debug_log(
        "loaded_sequences",
        data={"n_sequences": len(sequences)},
        location="web_app.py:analyze_fasta",
        hypothesis_id="H2",
        run_id="pre-fix",
    )
    if not sequences:
        raise HTTPException(status_code=400, detail="No sequences found in uploaded FASTA.")

    X = encode_sequences(sequences)
    _debug_log(
        "encoded_sequences",
        data={"shape": getattr(X, "shape", None)},
        location="web_app.py:analyze_fasta",
        hypothesis_id="H3",
        run_id="pre-fix",
    )

    # Option 1: classic pipeline (k-mers -> UMAP -> HDBSCAN)
    # Option 2: use deep autoencoder embeddings if available
    ae_path = BASE_DIR / "models" / "kmer_ae.pt"
    try:
        if use_deep_model and ae_path.exists():
            deep_embeddings = embed_with_autoencoder(X, ae_path)
            labels, embedding = cluster_sequences(deep_embeddings)
            _debug_log(
                "clustered_with_deep_embeddings",
                data={
                    "n_sequences": len(sequences),
                    "embedding_shape": getattr(embedding, "shape", None),
                },
                location="web_app.py:analyze_fasta",
                hypothesis_id="H4",
                run_id="pre-fix",
            )
        else:
            labels, embedding = cluster_sequences(X)
            _debug_log(
                "clustered_with_kmers",
                data={
                    "n_sequences": len(sequences),
                    "embedding_shape": getattr(embedding, "shape", None),
                },
                location="web_app.py:analyze_fasta",
                hypothesis_id="H4",
                run_id="pre-fix",
            )
    except Exception as exc:
        _debug_log(
            "clustering_failed",
            data={"error": str(exc)},
            location="web_app.py:analyze_fasta",
            hypothesis_id="H4",
            run_id="pre-fix",
        )
        raise

    metrics = summarise_biodiversity(labels)

    # Novelty / OOD scores in the embedding space
    novelty_scores = sequence_novelty_scores(embedding, labels)
    cluster_novelty = cluster_novelty_summary(embedding, labels, scores=novelty_scores)

    # Optional: BLAST-based taxonomic annotation per cluster
    # (returns empty dict if BLAST is not configured)
    taxo_tmp_dir = BASE_DIR / "taxonomy_tmp"
    cluster_taxonomy = annotate_clusters(
        sequences=sequences,
        labels=np.array(labels),
        embedding=np.array(embedding),
        cluster_novelty=cluster_novelty,
        tmp_dir=taxo_tmp_dir,
    )

    # Convert numpy arrays to plain Python lists for JSON
    embedding_list = embedding.tolist()
    labels_list = labels.tolist()

    num_clusters = int(metrics["richness"])
    noise_count = int((np.array(labels) == -1).sum())

    return {
        "n_sequences": len(sequences),
        "num_clusters": num_clusters,
        "noise_count": noise_count,
        "labels": labels_list,
        "embedding": embedding_list,
        "biodiversity": metrics,
        "novelty_scores": novelty_scores.tolist(),
        "cluster_novelty": cluster_novelty,
        "cluster_taxonomy": {
            str(cid): {
                "cluster_id": ann.cluster_id,
                "size": ann.size,
                "best_identity": ann.best_identity,
                "best_description": ann.best_description,
                "novelty_score": ann.novelty_score,
            }
            for cid, ann in cluster_taxonomy.items()
        },
        "used_deep_model": bool(use_deep_model and ae_path.exists()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)

