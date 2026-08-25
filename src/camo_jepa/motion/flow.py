"""Optical flow estimation, token encoding, and reconstruction modules for the motion branch."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class PrecomputedFlowReader(nn.Module):
    """Motion estimator that simply reads pre-computed Optical Flow (e.g., LiteFlowNet3).
    
    In this design, the Dataloader is responsible for loading the `.npy` files and
    passing them into the model via the `flows` argument.
    """

    def forward(self, images: torch.Tensor, flows: torch.Tensor | None = None) -> torch.Tensor:
        if flows is None:
            raise ValueError(
                "PrecomputedFlowReader requires pre-computed flows to be passed in! "
                "Ensure your Dataloader loads the LiteFlowNet3 .npy files and adds them "
                "to the FrameBatch."
            )
        if flows.ndim != 5:
            raise ValueError("flows must have shape [batch, time-1, channels, height, width]")
        return flows


class FlowTokenEncoder(nn.Module):
    """Core trainable encoder: projects 2-channel flow maps to spatial patches [B, T-1, N, 1024].

    Uses a single non-overlapping patch embedding (kernel=16, stride=16) matching V-JEPA2's
    visual tokeniser patch size so that motion and static tokens are spatially aligned.
    """

    def __init__(self, flow_channels: int = 2, token_dim: int = 1024) -> None:
        super().__init__()
        self.patch_size = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(flow_channels, token_dim, kernel_size=16, stride=16),
            nn.LayerNorm(token_dim),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5:
            raise ValueError("flow must have shape [batch, time, channels, height, width]")
        batch, frames, channels, height, width = flow.shape
        flow_flat = flow.reshape(batch * frames, channels, height, width)
        conv_layer, norm_layer = self.encoder[0], self.encoder[1]
        x = conv_layer(flow_flat)               # [B*(T-1), token_dim, H/16, W/16]
        x = x.flatten(2).permute(0, 2, 1)       # [B*(T-1), N, token_dim]
        x = norm_layer(x)                        # LayerNorm on token_dim
        return x.reshape(batch, frames, -1, x.size(-1))


class FlowDecoder(nn.Module):
    """Pre-training ONLY module: reconstructs flow map from dynamic_patches for self-supervised loss."""

    def __init__(self, token_dim: int = 1024, flow_channels: int = 2) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, flow_channels, kernel_size=4, stride=4),
        )

    def forward(
        self,
        dynamic_patches: torch.Tensor,
        spatial_grid: tuple[int, int] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Reconstruct flow map from patch tokens.

        Args:
            dynamic_patches: [B, T-1, N, D]
            spatial_grid: (H_p, W_p) number of patches per spatial axis. If None, inferred
                          from ``image_size`` or assumed square-ish from token count.
            image_size: (H, W) of the original flow map — used to infer spatial_grid when
                        spatial_grid is not supplied.
        """
        batch, frames, token_count, dim = dynamic_patches.shape
        if spatial_grid is None:
            if image_size is not None:
                h_grid = image_size[0] // 16
                w_grid = image_size[1] // 16
            else:
                # Fallback: assume 1:2 aspect ratio (256×512 default)
                import math
                w_grid = int(math.sqrt(token_count * 2))
                h_grid = token_count // w_grid
            spatial_grid = (h_grid, w_grid)
        h_grid, w_grid = spatial_grid
        x = dynamic_patches.view(batch * frames, h_grid, w_grid, dim).permute(0, 3, 1, 2)
        flow_recon = self.decoder(x)
        _, c, h, w = flow_recon.shape
        return flow_recon.view(batch, frames, c, h, w)


class MotionPretrainPipeline(nn.Module):
    """Autoencoder pipeline for the motion pre-training stage.

    Design:

        flows [B, T-1, 2, H, W]  (pre-computed by LiteFlowNet3)
          -> FlowTokenEncoder (trainable)  -> dynamic_patches [B, T-1, N, D]
          -> FlowDecoder      (trainable)  -> pred_flow [B, T-1, 2, H, W]
          -> L1 loss vs. gt_flow

    Only ``flow_encoder`` weights are carried forward to Phase 1.
    ``flow_decoder`` is discarded after pre-training.
    """

    def __init__(self, flow_channels: int = 2, token_dim: int = 1024) -> None:
        super().__init__()
        self.flow_estimator = PrecomputedFlowReader()
        self.flow_encoder = FlowTokenEncoder(flow_channels, token_dim)
        self.flow_decoder = FlowDecoder(token_dim, flow_channels)

    def forward(self, images: torch.Tensor, flows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one autoencoding forward pass.

        Args:
            images: [B, T, C, H, W] raw RGB frames.
            flows: [B, T-1, 2, H, W] pre-computed optical flow.

        Returns:
            pred_flow: [B, T-1, 2, H, W]
            loss: scalar L1 reconstruction loss
        """
        with torch.no_grad():
            gt_flow = self.flow_estimator(images, flows=flows)

        dynamic_patches = self.flow_encoder(gt_flow)
        image_size = (images.size(-2), images.size(-1))
        pred_flow = self.flow_decoder(dynamic_patches, image_size=image_size)

        loss = F.l1_loss(pred_flow, gt_flow)
        return pred_flow, loss