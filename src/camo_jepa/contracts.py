"""Typed tensor contracts shared across CaMo-JEPA stages."""

from dataclasses import dataclass

import torch


@dataclass
class FrameBatch:
    """A batch of temporally ordered CaMo image frames and ego motion.

    ``images`` has shape ``[B, T, C, H, W]`` where ``B`` is batch size,
    ``T`` is sequence/history length, ``C`` is normally RGB=3, and ``H, W``
    are the resized image dimensions. The CaMo loader returns ``float32``
    images in the ``[0, 1]`` range.

    ``ego_motion`` is optional and, when supplied, has shape ``[B, T, M]``.
    For ``dataset_camo``, ``M=2`` and values are
    ``[yaw_rate_radps, longitudinal_acceleration_mps2]``. Every image and
    ego-motion vector at time index ``t`` comes from the same source frame
    timestamp. These are observed vehicle-motion signals, not direct driving
    controls returned by Drive-JEPA.
    
    ``flows`` is optional and, when supplied, has shape ``[B, T-1, 2, H, W]``.
    It contains pre-computed Optical Flow (e.g. from LiteFlowNet3).
    """

    images: torch.Tensor
    ego_motion: torch.Tensor | None = None
    flows: torch.Tensor | None = None


    def validate(self) -> None:
        if self.images.ndim != 5:
            raise ValueError("images must have shape [batch, time, channels, height, width]")
        if self.ego_motion is not None and self.ego_motion.shape[:2] != self.images.shape[:2]:
            raise ValueError("ego_motion must match the batch and time dimensions")


@dataclass
class MotionBatch:
    flow: torch.Tensor
    tokens: torch.Tensor


@dataclass
class LatentBatch:
    static: torch.Tensor
    dynamic: torch.Tensor
    fused: torch.Tensor
    task: torch.Tensor | None = None
    exogenous: torch.Tensor | None = None
    confounder: torch.Tensor | None = None


@dataclass
class ModelOutput:
    """Canonical forward output for ``CaMoJEPAPipeline``.

    Shapes:
    - ``z_pred``: ``[B, M, D]``
    - ``target_patches``: ``[B, M, D]``
    - ``z_task``: ``[B, (T-1) * N, D]``
    - ``z_exogenous``: ``[B, (T-1) * N, D]``
    - ``U``: ``[B, U_dim]``
    - ``fused_features``: ``[B, T-1, N, D]``
    - ``static_features``: ``[B, T-1, N, D]``
    """

    z_pred: torch.Tensor
    target_patches: torch.Tensor
    z_task: torch.Tensor
    z_exogenous: torch.Tensor
    U: torch.Tensor
    fused_features: torch.Tensor
    static_features: torch.Tensor
    losses: dict[str, torch.Tensor] | None = None
    mask_indices: torch.Tensor | None = None
