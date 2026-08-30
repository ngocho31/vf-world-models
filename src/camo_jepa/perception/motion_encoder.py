"""FlowFormer++ Estimator and Motion Patch Token Encoder."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip DDP and wrapper prefixes ('module.', 'backbone.', 'model.', 'encoder.')."""
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key
        while any(clean_key.startswith(p) for p in ["module.", "backbone.", "model.", "encoder."]):
            for prefix in ["module.", "backbone.", "model.", "encoder."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
        normalized[clean_key] = value
    return normalized


class FlowFormerPlusPlusEstimator(nn.Module):
    """FlowFormer++ optical flow estimator dynamically imported from source root."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        flowformer_root: str | Path,
        config_name: str = "submissions",
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        flowformer_root = Path(flowformer_root).expanduser().resolve()
        self._flowformer_root = flowformer_root

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"FlowFormer++ checkpoint does not exist: {checkpoint_path}")

        # Import FlowFormer++ modules dynamically from the source root.
        build_flowformer, get_cfg = self._import_flowformer_modules(flowformer_root, config_name)

        # Fetch the configuration and build the FlowFormer++ backbone.
        cfg = get_cfg()
        self.backbone = build_flowformer(cfg)

        # Load the checkpoint into the backbone.
        self._load_checkpoint(checkpoint_path)

    @staticmethod
    def _import_flowformer_modules(flowformer_root: Path, config_name: str):
        if not flowformer_root.is_dir():
            raise FileNotFoundError(f"FlowFormer++ source root does not exist: {flowformer_root}")

        # --- Monkey patch timm for FlowFormer++ compatibility ---
        import timm.models.helpers
        import timm.models.layers
        import timm.layers.activations as timm_activations

        # 1. Ensure timm.models.layers has the activations attribute.
        if not hasattr(timm.models.layers, "activations"):
            timm.models.layers.activations = timm_activations

        # 2. Ensure timm.models.helpers has the overlay_external_default_cfg function.
        if not hasattr(timm.models.helpers, "overlay_external_default_cfg"):
            def overlay_external_default_cfg(default_cfg, model_cfg):
                if model_cfg is None:
                    return default_cfg
                for k, v in model_cfg.items():
                    if v is not None:
                        default_cfg[k] = v
                return default_cfg

            timm.models.helpers.overlay_external_default_cfg = overlay_external_default_cfg
        # ----------------------------------------------------------------------

        root_str = str(flowformer_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            build_flowformer = importlib.import_module("core.FlowFormer").build_flowformer
            get_cfg = importlib.import_module(f"configs.{config_name}").get_cfg
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Failed to import FlowFormer++ modules from {flowformer_root}. "
                f"Original error: {e}"
            ) from e
        return build_flowformer, get_cfg

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load and inject compatible state dict weights into backbone."""
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ["model", "state_dict", "flow_encoder", "model_state_dict"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        normalized = _normalize_state_dict(state_dict)
        expected = self.backbone.state_dict()

        compatible = {
            key: value
            for key, value in normalized.items()
            if key in expected and expected[key].shape == value.shape
        }

        if not compatible:
            raise ValueError(f"No compatible weights found in checkpoint: {checkpoint_path}")

        self.backbone.load_state_dict(compatible, strict=False)
        print(f"Loaded compatible weights from checkpoint: {checkpoint_path}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return 2-channel optical flow maps [B, T-1, 2, H, W] for consecutive frames."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images shape [B, T, 3, H, W], got {tuple(images.shape)}")
        batch, frames, channels, height, width = images.shape
        if frames < 2:
            raise ValueError("images must contain at least two consecutive frames")

        # FlowFormer++ expects images in the range [0, 255]
        if images.max() <= 1.0:
            images = images * 255.0

        img1 = images[:, :-1].reshape(-1, channels, height, width)
        img2 = images[:, 1:].reshape(-1, channels, height, width)
        output = self.backbone(img1, img2)

        flow = output
        while isinstance(flow, (tuple, list)):
            if isinstance(flow[0], (tuple, list)):
                flow = flow[0]
            else:
                # Take the last element if the flow is a list/tuple of multiple outputs.
                flow = flow[-1]

        if not isinstance(flow, torch.Tensor):
            raise TypeError(f"FlowFormer++ output format not recognized, got {type(flow)}")

        _, channels, h_out, w_out = flow.shape
        return flow.reshape(batch, frames - 1, channels, h_out, w_out)


class FlowTokenEncoder(nn.Module):
    """Projects 2-channel flow maps [B, T-1, 2, H, W] to patch tokens [B, T-1, N, D]."""

    def __init__(
        self,
        flow_channels: int = 2,
        token_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = 16
        self.proj = nn.Conv2d(flow_channels, token_dim, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Encode 2-channel flow maps into ViT-compatible patch tokens."""
        if flow.ndim != 5:
            raise ValueError("flow must have shape [batch, time, channels, height, width]")

        batch, frames, channels, height, width = flow.shape
        flow_flat = flow.reshape(batch * frames, channels, height, width)
        x = self.proj(flow_flat)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)

        return x.reshape(batch, frames, -1, x.size(-1))


def build_motion_encoders(
    flowformer_checkpoint_path: str | Path,
    flowformer_root: str | Path,
    token_dim: int = 1024,
    flow_channels: int = 2,
) -> tuple[FlowFormerPlusPlusEstimator, FlowTokenEncoder]:
    """Build optical flow estimator and motion token encoder components."""
    estimator = FlowFormerPlusPlusEstimator(
        checkpoint_path=flowformer_checkpoint_path,
        flowformer_root=flowformer_root,
    )
    token_encoder = FlowTokenEncoder(
        flow_channels=flow_channels,
        token_dim=token_dim,
    )
    return estimator, token_encoder