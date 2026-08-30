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
    """

    images: torch.Tensor
    ego_motion: torch.Tensor | None = None

    def validate(self) -> None:
        if self.images.ndim != 5:
            raise ValueError("images must have shape [batch, time, channels, height, width]")
        if self.ego_motion is not None and self.ego_motion.shape[:2] != self.images.shape[:2]:
            raise ValueError("ego_motion must match the batch and time dimensions")


@dataclass
class ModelOutput:
    """Canonical output for CaMoJEPAPipeline supporting training, evaluation, and ablation studies.

    Shapes:
    - ``z_pred``: [B, M, D] - Predicted latent representations
    - ``target_patches``: [B, M, D] - Target patch representations
    - ``static_features``: [B, T-1, N, D] - Raw static ViT features
    - ``fused_features``: [B, T-1, N, D] - Fused multimodal features
    - ``z_task``: [B, (T-1)*N, D] or [B, T-1, N, D] - Task-specific representations
    - ``dynamic_features``: Optional [B, T-1, N, D] - Motion flow tokens
    - ``z_exogenous``: Optional [B, (T-1)*N, D] - Exogenous features
    - ``P_task``: Optional [D, D_task] - Task projection matrix for do(a_t) interventions
    - ``P_exogenous``: Optional [D, D_exo] - Exogenous projection matrix
    - ``U``: Optional [B, U_dim] - Confounder context vector
    - ``mask_indices``: Optional [B, M] - Indices of masked target patches
    - ``context_indices``: Optional [B, N_ctx] - Indices of visible context patches
    - ``losses``: Optional dict of computed losses
    """

    # 1. Standard Prediction Outputs
    z_pred: torch.Tensor
    target_patches: torch.Tensor

    # 2. Intermediate Representations (For Feature Probing & Modality Ablation)
    static_features: torch.Tensor
    fused_features: torch.Tensor
    z_task: torch.Tensor
    dynamic_features: torch.Tensor | None = None

    # 3. Disentanglement & Causal Outputs (For Causal Ablation & Interventions)
    z_exogenous: torch.Tensor | None = None
    P_task: torch.Tensor | None = None
    P_exogenous: torch.Tensor | None = None
    U: torch.Tensor | None = None

    # 4. Masking & Diagnostic Tools
    mask_indices: torch.Tensor | None = None
    context_indices: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None
