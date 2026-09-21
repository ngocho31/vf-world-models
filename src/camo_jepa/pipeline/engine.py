"""Training engine execution logic for CaMo-JEPA.

OPTIMIZATION NOTES (2026-09-14):
- [OPT-1] Added Automatic Mixed Precision (AMP) with bfloat16 via torch.autocast.
  A100 GPUs have dedicated Tensor Cores for bfloat16 which doubles throughput.
  GradScaler is passed in from cli.py to handle gradient scaling safely.
- [OPT-2] Made DataParallel-safe: unwrap model.module before calling
  update_target_encoder() so EMA update works correctly on multi-GPU setups.
"""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import FrameBatch, ModelOutput
from .phase1 import CaMoJEPAPipeline


def train_step(
    model: CaMoJEPAPipeline | nn.DataParallel | nn.parallel.DistributedDataParallel,
    batch: FrameBatch,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> ModelOutput:
    """Run forward pass, compute total loss, backpropagate, step optimizer, and update target encoder via EMA.

    Args:
        model: The CaMoJEPAPipeline (or DataParallel/DistributedDataParallel-wrapped version).
        batch: A FrameBatch of images and metadata.
        optimizer: The AdamW optimizer.
        loss_fn: Optional loss function override. Defaults to model.loss_fn.
        scaler: Optional GradScaler for AMP mixed-precision training.
    """
    # [OPT-2] Unwrap DataParallel / DistributedDataParallel to access the underlying model's methods
    base_model = getattr(model, "module", model)
    base_model.train()

    active_loss_fn = loss_fn if loss_fn is not None else base_model.loss_fn

    # [OPT-1] Forward pass under bfloat16 autocast for ~2x speedup on A100
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        output = model(batch)
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

    if scaler is not None:
        # [OPT-1] Scale gradients to prevent underflow in fp16/bf16
        scaler.scale(total_loss).backward()
        # Unscale before checking gradients or stepping
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()

    # [OPT-2] EMA update must run on the base model, not the DataParallel/DDP wrapper
    base_model.update_target_encoder()

    return output