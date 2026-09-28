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
    log_gate_gradients: bool = False,
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

    if log_gate_gradients:
        gate = base_model.fusion.gate
        
        def gradient_value(loss: torch.Tensor) -> float:
            if not loss.requires_grad or not gate.requires_grad:
                return 0.0
            gradient = torch.autograd.grad(
                loss,
                gate,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return float(gradient.detach().cpu().item()) if gradient is not None else 0.0
            
        raw_jepa_gradient = gradient_value(losses["jepa"])
        raw_orthogonality_gradient = gradient_value(losses["orthogonality"])
        raw_reconstruction_gradient = gradient_value(losses["reconstruction"])
        
        jepa_gradient = active_loss_fn.lambda_jepa * raw_jepa_gradient
        orthogonality_gradient = active_loss_fn.lambda_orth * raw_orthogonality_gradient
        reconstruction_gradient = active_loss_fn.lambda_recon * raw_reconstruction_gradient
        
        output.diagnostics = {
            "gate_value": float(gate.detach().cpu().item()),
            "gate_tanh": float(torch.tanh(gate.detach()).cpu().item()),
            "d_jepa_d_gate": raw_jepa_gradient,
            "d_orthogonality_d_gate": raw_orthogonality_gradient,
            "d_reconstruction_d_gate": raw_reconstruction_gradient,
            "lambda_jepa_d_jepa_d_gate": jepa_gradient,
            "lambda_orth_d_orthogonality_d_gate": orthogonality_gradient,
            "lambda_reconstruction_d_reconstruction_d_gate": reconstruction_gradient,
            "weighted_d_jepa_d_gate": jepa_gradient,
            "weighted_d_orthogonality_d_gate": orthogonality_gradient,
            "weighted_d_reconstruction_d_gate": reconstruction_gradient,
            "weighted_gradients_sum": jepa_gradient + orthogonality_gradient + reconstruction_gradient,
            "d_total_d_gate": gradient_value(total_loss),
        }

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