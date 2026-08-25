"""Confounder sequence encoder for CaMo-JEPA."""

from __future__ import annotations

import torch
from torch import nn

class ConfounderGRU(nn.Module):
    """Compress temporal fused history into confounder vector ``U``."""

    def __init__(self, latent_dim: int, confounder_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=confounder_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            # Spatial mean-pool: [B, T, N_patches, D] → [B, T, D]
            history = history.mean(dim=2)
        if history.ndim != 3:
            raise ValueError(
                "history must have shape [batch, time, latent_dim] "
                "or [batch, time, patches, latent_dim]"
            )
        if history.shape[-1] != self.latent_dim:
            raise ValueError(
                f"ConfounderGRU expects last dim = {self.latent_dim}; "
                f"got {tuple(history.shape)}"
            )

        _, hidden = self.gru(history)
        return hidden[-1]  
