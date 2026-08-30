"""Independent causal latent stages."""

import math
from typing import Optional
import torch
from torch import nn


class LatentFactorizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        task_dim: Optional[int] = None,
        enforce_orthogonality: bool = True
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.enforce_orthogonality = enforce_orthogonality

        if self.enforce_orthogonality:
            self.task_dim = task_dim if task_dim is not None else latent_dim // 2
            if not (0 < self.task_dim < latent_dim):
                raise ValueError(
                    f"task_dim must be in (0, latent_dim); got task_dim={self.task_dim}, latent_dim={latent_dim}"
                )
            self.raw_projection = nn.Parameter(torch.randn(latent_dim, self.task_dim) * (1.0 / math.sqrt(latent_dim)))
        else:
            self.task_dim = None
            self.p_task_linear = nn.Linear(latent_dim, latent_dim, bias=False)

    def _compute_p_task(self, dtype: torch.dtype) -> torch.Tensor:
        if self.enforce_orthogonality:
            # Compute the orthogonal projector using the QR decomposition in FP32 for numerical stability
            q, _ = torch.linalg.qr(self.raw_projection.to(torch.float32), mode="reduced")
            p_task = q @ q.transpose(0, 1)
            return p_task.to(dtype)
        return self.p_task_linear.weight.to(dtype)

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            fused (torch.Tensor): Fused latent representation, shape [B, T-1, N, D] or [B, N, D].
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Task-specific latent representation, exogenous latent representation, task projection matrix, and exogenous projection matrix.
        """
        p_task = self._compute_p_task(fused.dtype)
        identity = torch.eye(self.latent_dim, device=fused.device, dtype=fused.dtype)
        p_exogenous = identity - p_task

        z_task = fused @ p_task.transpose(0, 1)
        z_exo = fused @ p_exogenous.transpose(0, 1)

        return z_task, z_exo, p_task, p_exogenous
