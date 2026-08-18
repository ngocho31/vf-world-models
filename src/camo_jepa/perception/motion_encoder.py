"""Motion encoder adapter: loads pre-trained FlowTokenEncoder weights for Phase 1 fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..motion import FlowTokenEncoder, FrameDifferenceFlow


class MotionEncoderAdapter(nn.Module):
    """Wraps a pre-trained FlowTokenEncoder as a trainable adapter for the Phase 1 pipeline.

    Responsibilities:
    - Run the frozen flow estimator (no-grad) to produce flow maps.
    - Forward flow maps through the adapter (FlowTokenEncoder) which is fine-tuned end-to-end
      with the rest of the CaMo-JEPA pipeline.

    Weight origin:
        Pre-trained by ``MotionPretrainPipeline`` in ``motion/flow.py``, saved as a bare
        ``FlowTokenEncoder.state_dict()``.  Loaded here via ``build_motion_encoder_adapter``.

    Args:
        flow_channels: Number of input channels for the flow map (default 2: dx, dy).
        token_dim: Dimensionality of output patch tokens (must match ViT encoder dim).
        freeze: If True, freeze the adapter weights (useful for ablations).
    """

    def __init__(
        self,
        flow_channels: int = 2,
        token_dim: int = 1024,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.flow_estimator = FrameDifferenceFlow()   # always frozen — no learnable params
        self.flow_encoder = FlowTokenEncoder(flow_channels, token_dim)

        if freeze:
            self.flow_encoder.requires_grad_(False)
            self.flow_encoder.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode consecutive-frame flow into patch tokens.

        Args:
            images: [B, T, C, H, W] raw RGB frames.

        Returns:
            dynamic_patches: [B, T-1, N, token_dim]
        """
        with torch.no_grad():
            gt_flow = self.flow_estimator(images)   # [B, T-1, 2, H, W]
        return self.flow_encoder(gt_flow)            # [B, T-1, N, D]


def build_motion_encoder_adapter(
    checkpoint_path: str | Path | None = None,
    flow_channels: int = 2,
    token_dim: int = 1024,
    freeze: bool = False,
) -> MotionEncoderAdapter:
    """Instantiate a ``MotionEncoderAdapter`` and optionally load pre-trained weights.

    The checkpoint is the bare ``FlowTokenEncoder.state_dict()`` saved after
    ``MotionPretrainPipeline`` training (see ``motion/flow.py``).

    Args:
        checkpoint_path: Path to a ``.pt`` file containing ``FlowTokenEncoder`` weights.
            Accepts both a bare state dict *and* a keyed payload
            ``{"flow_encoder": state_dict, ...}`` from a full CaMo-JEPA checkpoint.
        flow_channels: Passed to ``MotionEncoderAdapter``.
        token_dim: Passed to ``MotionEncoderAdapter``.
        freeze: If True, freeze ``flow_encoder`` after loading.

    Returns:
        A ``MotionEncoderAdapter`` ready for Phase 1 training.
    """
    adapter = MotionEncoderAdapter(flow_channels=flow_channels, token_dim=token_dim, freeze=freeze)

    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu")
            # Support both bare state dict and keyed payload
            state_dict = checkpoint.get("flow_encoder", checkpoint)
            adapter.flow_encoder.load_state_dict(state_dict)
            print(f"[MotionEncoderAdapter] Loaded pre-trained weights from {path}")
        else:
            print(f"[MotionEncoderAdapter] Checkpoint not found at {path}, using random weights.")

    return adapter