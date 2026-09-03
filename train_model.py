"""
CLI helper to train the k-mer autoencoder on a given FASTA file.

Example:
    python train_model.py data/reads.fasta models/kmer_ae.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_fasta
from kmer_encoder import encode_sequences
from models import train_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Train k-mer autoencoder on eDNA reads.")
    parser.add_argument("fasta", type=str, help="Input FASTA file with reads.")
    parser.add_argument(
        "output",
        type=str,
        help="Path to save trained model weights, e.g. models/kmer_ae.pt",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    sequences = load_fasta(args.fasta)
    if not sequences:
        raise SystemExit("No sequences found; aborting.")

    X = encode_sequences(sequences)
    print(f"Encoded {len(sequences)} sequences with feature dim={X.shape[1]}")

    train_autoencoder(
        X,
        model_path=args.output,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

