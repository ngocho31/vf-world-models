"""Non-training metrics for reports."""

import torch
import torch.nn.functional as F


def latent_drift(latent: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mean squared drift from the frozen reference representation."""
    return F.mse_loss(latent, reference)


def cosine_similarity(latent: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(latent, reference, dim=-1).mean()
