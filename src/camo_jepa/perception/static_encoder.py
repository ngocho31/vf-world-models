"""Frozen Drive-JEPA V-JEPA2 ViT-L context and target encoders.

This module is a thin wrapper around the V-JEPA2 `vit_large` implementation
and `robust_checkpoint_loader`, returning full patch tokens for each
overlapping two-frame clip.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch import nn


class DriveJEPAViTLEncoder(nn.Module):
    """V-JEPA2 ViT-L encoder loaded from the Drive-JEPA checkpoint state dict."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        vjepa2_root: str | Path,
        image_size: tuple[int, int],
        checkpoint_key: str = "encoder",
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        vjepa2_root = Path(vjepa2_root).expanduser().resolve()
        self._vjepa2_root = vjepa2_root
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Drive-JEPA ViT-L checkpoint does not exist: {checkpoint_path}")

        vit_large = self._import_vit_large(vjepa2_root)
        width, height = image_size
        self.backbone = vit_large(
            img_size=(height, width),
            num_frames=2,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
            handle_nonsquare_inputs=True,
        )
        self._load_checkpoint(checkpoint_path, checkpoint_key)

    @staticmethod
    def _import_vit_large(vjepa2_root: Path):
        source_root = vjepa2_root / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(f"V-JEPA2 source directory does not exist: {source_root}")
        import src as workspace_src

        source_root_string = str(source_root)
        if source_root_string not in workspace_src.__path__:
            workspace_src.__path__.append(source_root_string)
        return importlib.import_module("src.models.vision_transformer").vit_large

    @staticmethod
    def _import_checkpoint_loader(vjepa2_root: Path):
        source_root = vjepa2_root / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(f"V-JEPA2 source directory does not exist: {source_root}")
        import src as workspace_src

        source_root_string = str(source_root)
        if source_root_string not in workspace_src.__path__:
            workspace_src.__path__.append(source_root_string)
        return importlib.import_module("src.utils.checkpoint_loader").robust_checkpoint_loader

    def _load_checkpoint(self, checkpoint_path: Path, checkpoint_key: str) -> None:
        checkpoint_loader = self._import_checkpoint_loader(self._vjepa2_root)
        checkpoint = checkpoint_loader(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get(checkpoint_key)
        if not isinstance(state_dict, dict):
            raise ValueError(f"checkpoint has no {checkpoint_key} state dict: {checkpoint_path}")
        normalized = _normalize_state_dict(state_dict)
        expected = self.backbone.state_dict()
        compatible = {
            key: value
            for key, value in normalized.items()
            if key in expected and expected[key].shape == value.shape
        }
        missing, _ = self.backbone.load_state_dict(compatible, strict=False)
        if "patch_embed.proj.weight" in missing:
            raise ValueError(f"checkpoint is incompatible with V-JEPA ViT-L: {checkpoint_path}")
        del checkpoint, state_dict, normalized, compatible, expected

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens ``[B, T-1, N, D]`` for consecutive clips ``[t, t + 1]``."""
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, time, channels, height, width]")
        batch, frames, channels, height, width = images.shape
        if frames < 2:
            raise ValueError("images must contain at least two consecutive frames")
        clips = torch.stack((images[:, :-1], images[:, 1:]), dim=2)
        clips = clips.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * (frames - 1), channels, 2, height, width
        )
        tokens = self.backbone(clips)
        return tokens.reshape(batch, frames - 1, tokens.size(1), tokens.size(2))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for `forward_tokens` to preserve patch-token semantics."""
        return self.forward_tokens(images)


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("backbone."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]
        normalized[key] = value
    return normalized


def build_vit_encoders(
    checkpoint_path: str | Path,
    vjepa2_root: str | Path,
    image_size: tuple[int, int],
) -> tuple[DriveJEPAViTLEncoder, DriveJEPAViTLEncoder]:
    """Build frozen context and target ViT-L encoders from one Drive-JEPA checkpoint."""
    context_encoder = DriveJEPAViTLEncoder(
        checkpoint_path=checkpoint_path,
        vjepa2_root=vjepa2_root,
        image_size=image_size,
        checkpoint_key="encoder",
    )
    target_encoder = DriveJEPAViTLEncoder(
        checkpoint_path=checkpoint_path,
        vjepa2_root=vjepa2_root,
        image_size=image_size,
        checkpoint_key="target_encoder",
    )
    return context_encoder, target_encoder

