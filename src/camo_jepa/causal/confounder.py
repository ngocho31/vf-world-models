"""Confounder sequence encoder for CaMo-JEPA."""

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

    def forward(self, fused_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_global (torch.Tensor): Global fused representation after Spatial Mean-Pool,
            shape [B, T-1, latent_dim] (Strictly 3D).
        Returns:
            torch.Tensor: Confounder Context Vector U_t, shape [B, confounder_dim].
        """
        if fused_global.ndim != 3:
            raise ValueError(
                f"ConfounderGRU expects strictly 3D tensor [batch, time, latent_dim]; "
                f"got shape {tuple(fused_global.shape)}"
            )

        if fused_global.shape[-1] != self.latent_dim:
            raise ValueError(
                f"ConfounderGRU expects the last dimension to be {self.latent_dim}; "
                f"got {fused_global.shape[-1]}"
            )

        _, hidden = self.gru(fused_global)
        return hidden[-1] # Output shape: [B, confounder_dim] (e.g., [B, 128])
