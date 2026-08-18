"""Confounder sequence encoder for CaMo-JEPA."""

import torch
from torch import nn


class ConfounderGRU(nn.Module):
    """Compress temporal fused history into confounder vector ``U``."""

    def __init__(self, latent_dim: int, confounder_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(latent_dim, confounder_dim, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 4:
            # Spatial mean-pool: [B, T, N_patches, D] → [B, T, D]
            history = history.mean(dim=2)
        if history.ndim != 3:
            raise ValueError(
                "history must have shape [batch, time, latent_dim] "
                "or [batch, time, patches, latent_dim]"
            )
        _, hidden = self.gru(history)
        return hidden[-1]
