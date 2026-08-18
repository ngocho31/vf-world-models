"""Static and motion feature fusion."""

import torch
from torch import nn


class ResidualFusion(nn.Module):
    def __init__(self, latent_dim: int, dynamic_dim: int | None = None, fusion_scale: float = 1.0) -> None:
        super().__init__()
        self.dynamic_projection = (
            nn.Identity() if dynamic_dim in (None, latent_dim) else nn.Linear(dynamic_dim, latent_dim)
        )
        self.gate = nn.Parameter(torch.zeros(latent_dim))
        self.fusion_scale = fusion_scale

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        if static.shape[:-1] != dynamic.shape[:-1]:
            raise ValueError("static and dynamic latents must share batch and time dimensions")
        dynamic = self.dynamic_projection(dynamic)
        return static + self.fusion_scale * torch.tanh(self.gate) * dynamic