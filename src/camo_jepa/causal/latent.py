"""Independent causal latent stages."""

import torch
from torch import nn


class LatentFactorizer(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.task_projection = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task = self.task_projection(fused)
        exogenous = fused - task
        return task, exogenous
