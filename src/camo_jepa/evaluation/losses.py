"""Differentiable metrics used by the experiments."""

import torch
import torch.nn.functional as F
from torch import nn


def compute_jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Normalized Smooth L1 loss between predicted and target tokens."""
    z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
    z_target = F.layer_norm(z_target, (z_target.size(-1),))
    return F.smooth_l1_loss(z_pred, z_target)


def compute_reconstruction_loss(fused: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(fused, static, dim=-1)).mean()


def compute_orthogonality_loss(z_task: torch.Tensor, z_exogenous: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    D = z_task.shape[-1]

    z_task_flat = z_task.reshape(-1, D)
    z_exogenous_flat = z_exogenous.reshape(-1, D)
    n_samples = z_task_flat.shape[0]

    if normalize:
        z_task_flat = F.normalize(z_task_flat, p=2, dim=-1)
        z_exogenous_flat = F.normalize(z_exogenous_flat, p=2, dim=-1)

    gram_matrix = (z_task_flat.transpose(0, 1) @ z_exogenous_flat) / n_samples

    return (gram_matrix ** 2).sum()


class CaMoJEPALoss(nn.Module):
    def __init__(self, lambda_jepa: float, lambda_orth: float, lambda_recon: float) -> None:
        super().__init__()
        self.lambda_jepa = lambda_jepa
        self.lambda_orth = lambda_orth
        self.lambda_recon = lambda_recon

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        z_task: torch.Tensor,
        z_exogenous: torch.Tensor,
        fused: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 1. JEPA Prediction Loss
        l_jepa = compute_jepa_loss(z_pred, z_target)

        # 2. Orthogonality Loss (if factorizer branch is enabled)
        if z_exogenous is not None:
            l_orth = compute_orthogonality_loss(z_task, z_exogenous)
        else:
            l_orth = torch.tensor(0.0, device=z_pred.device)

        # 3. Reconstruction Loss (if motion branch is enabled)
        if fused is static:
            l_recon = torch.tensor(0.0, device=z_pred.device)
        else:
            l_recon = compute_reconstruction_loss(fused, static)

        losses = {
            "jepa": l_jepa,
            "orthogonality": l_orth,
            "reconstruction": l_recon,
        }
        losses["total"] = (
            self.lambda_jepa * losses["jepa"]
            + self.lambda_orth * losses["orthogonality"]
            + self.lambda_recon * losses["reconstruction"]
        )
        return losses