"""Checkpoint persistence for trainable CaMo-JEPA modules."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save trainable CaMo-JEPA modules without frozen ViT encoders."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "flow_encoder": model.flow_encoder.state_dict(),
        "fusion": model.fusion.state_dict(),
        "factorizer": model.factorizer.state_dict(),
        "confounder": model.confounder.state_dict(),
        "predictor": model.predictor.state_dict(),
    }
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
    """Load trainable CaMo-JEPA modules from a saved checkpoint payload."""
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    model.flow_encoder.load_state_dict(checkpoint["flow_encoder"], strict=strict)
    model.fusion.load_state_dict(checkpoint["fusion"], strict=strict)
    model.factorizer.load_state_dict(checkpoint["factorizer"], strict=strict)
    model.confounder.load_state_dict(checkpoint["confounder"], strict=strict)
    model.predictor.load_state_dict(checkpoint["predictor"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint