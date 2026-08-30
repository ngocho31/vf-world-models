"""Checkpoint persistence for trainable CaMo-JEPA modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save trainable CaMo-JEPA modules (Flow Encoder, Fusion, Factorizer, Confounder, Predictor).

    Frozen ViT encoders (context/target) and frozen Flow Estimator are excluded.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
    }

    trainable_modules = [
        "flow_encoder",
        "fusion",
        "factorizer",
        "confounder",
        "predictor",
    ]
    for module_name in trainable_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None and isinstance(module, nn.Module):
                payload[module_name] = module.state_dict()

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, object]:
    """Load checkpoint safely across different ablation configurations."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"[Checkpoint] File not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    trainable_modules = [
        "flow_encoder",
        "fusion",
        "factorizer",
        "confounder",
        "predictor",
    ]

    for module_name in trainable_modules:
        if module_name in checkpoint and hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None and isinstance(module, nn.Module):
                module.load_state_dict(checkpoint[module_name], strict=strict)

    epoch = checkpoint["epoch"]
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint, epoch, optimizer