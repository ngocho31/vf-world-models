"""Training engine execution logic for CaMo-JEPA."""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import FrameBatch, ModelOutput
from .phase1 import CaMoJEPAPipeline


def train_step(
    model: CaMoJEPAPipeline,
    batch: FrameBatch,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module | None = None,
) -> ModelOutput:
    """Run forward pass, compute total loss, backpropagate, step optimizer, and update target encoder via EMA."""
    model.train()
    output = model(batch)

    active_loss_fn = loss_fn if loss_fn is not None else model.loss_fn
    losses = active_loss_fn(
        z_pred=output.z_pred,
        z_target=output.target_patches,
        z_task=output.z_task,
        z_exogenous=output.z_exogenous,
        fused=output.fused_features,
        static=output.static_features,
    )
    total_loss = losses["total"]
    output.losses = losses

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    model.update_target_encoder()

    return output