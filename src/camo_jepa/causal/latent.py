"""Independent causal latent stages."""

from __future__ import annotations

import torch
from torch import nn

import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class FactorizerOutput:
    z_task: torch.Tensor
    z_exo: torch.Tensor
    p_task: torch.Tensor
    p_exo: torch.Tensor

class LatentFactorizer(nn.Module):
    def __init__(self, latent_dim: int, task_dim: Optional[int] = None, enforce_orthogonal_projector: bool = True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.enforce_orthogonal_projector = enforce_orthogonal_projector
        self.register_buffer("identity", torch.eye(latent_dim), persistent=False)

        if enforce_orthogonal_projector:
            self.task_dim = task_dim if task_dim is not None else latent_dim // 2
            if not (0 < self.task_dim < latent_dim):
                raise ValueError(f"task_dim phải nằm trong (0, {latent_dim})")
            self.raw_basis = nn.Parameter(
                torch.randn(latent_dim, self.task_dim) * (1.0 / math.sqrt(latent_dim))
            )
        else:
            self.task_dim = None
            self.p_task_linear = nn.Linear(latent_dim, latent_dim, bias=False)

    def _compute_p_task(self) -> torch.Tensor:
        if self.enforce_orthogonal_projector:
            q, _ = torch.linalg.qr(self.raw_basis, mode="reduced")
            p_task = q @ q.transpose(0, 1)
            return p_task
        else:
            return self.p_task_linear.weight

    def forward(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused: tensor (B, ..., D) hoặc (B, T, N, D)
        Returns:
            z_task, z_exo: cùng shape với fused
        """
        p_task = self._compute_p_task()
        p_exo = self.identity - p_task

        z_task = fused @ p_task.transpose(0, 1)
        z_exo = fused @ p_exo.transpose(0, 1)
        return z_task, z_exo