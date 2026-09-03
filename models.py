"""
Deep sequence representation models for eDNA.

This module currently implements a simple k-mer autoencoder in PyTorch.
It learns a compact latent embedding from k-mer frequency vectors
produced by `kmer_encoder.encode_sequences`.

Usage pattern:
    - Train once offline with `train_autoencoder(...)`
    - At inference time, call `embed_with_autoencoder(...)` to obtain
      low-dimensional embeddings for clustering / novelty detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AEConfig:
    input_dim: int
    latent_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1


class KmerAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
            nn.Sigmoid(),  # k-mer frequencies are in [0, 1] after normalisation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)


def _to_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    X: np.ndarray,
    model_path: str | Path,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
) -> str:
    """
    Train an autoencoder on k-mer vectors and save weights.

    This is intended to be run offline (CLI / notebook), not per web request.
    """
    device = _to_device()
    X = X.astype(np.float32)
    input_dim = X.shape[1]

    cfg = AEConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    model = KmerAutoencoder(cfg).to(device)

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optim.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optim.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[AE] epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, model_path)
    print(f"[AE] saved model to {model_path}")
    return str(model_path)


def embed_with_autoencoder(
    X: np.ndarray,
    model_path: str | Path,
) -> np.ndarray:
    """
    Load a trained autoencoder and return latent embeddings for X.
    """
    device = _to_device()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Autoencoder weights not found at: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    cfg = AEConfig(**ckpt["config"])
    model = KmerAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = X.astype(np.float32)
    with torch.no_grad():
        x_tensor = torch.from_numpy(X).to(device)
        z = model.encode(x_tensor)
        return z.cpu().numpy()

