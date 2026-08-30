"""Causal predictor integrating the V-JEPA2 predictor and confounder conditioning.

This module reuses the upstream V-JEPA2 `vit_predictor` implementation and
its checkpoint loader, then adds CaMo-JEPA's confounder injection in front of
the predictor call.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch import nn

from .static_encoder import _normalize_state_dict


class CausalPredictor(nn.Module):
    """Combines the upstream V-JEPA2 patch predictor with confounder `U`."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        vjepa2_root: str | Path,
        image_size: tuple[int, int],
        num_transitions: int = 1,
        latent_dim: int = 1024,
        confounder_dim: int = 128,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 24,
        predictor_num_heads: int = 12,
        num_mask_tokens: int = 10,
        mode: str = "add",
    ) -> None:
        super().__init__()
        if mode not in {"add", "concat"}:
            raise ValueError("mode must be one of {'add', 'concat'}")

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        vjepa2_root = Path(vjepa2_root).expanduser().resolve()
        self._vjepa2_root = vjepa2_root
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Drive-JEPA predictor checkpoint does not exist: {checkpoint_path}")

        # 1. Projection layers for Confounder U
        self.mode = mode
        self.u_projection = nn.Linear(confounder_dim, latent_dim)
        self.concat_projection = (
            nn.Linear(latent_dim * 2, latent_dim) if mode == "concat" else nn.Identity()
        )

        # 2. Build & load base V-JEPA predictor backbone
        # tubelet_size=2 is the V-JEPA2 default; num_frames must equal
        # num_transitions * tubelet_size so that num_patches = num_transitions * H_p * W_p
        # matches the flattened (T-1)*N_patches token sequence.
        _tubelet_size = 2
        predictor_factory = self._import_predictor(vjepa2_root)
        width, height = image_size
        self.predictor = predictor_factory(
            img_size=(height, width),
            patch_size=16,
            num_frames=num_transitions * _tubelet_size,
            tubelet_size=_tubelet_size,
            embed_dim=1024,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            use_mask_tokens=True,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=True,
            uniform_power=True,
            use_rope=True,
        )
        self._load_checkpoint(checkpoint_path)

    @staticmethod
    def _import_predictor(vjepa2_root: Path):
        source_root = vjepa2_root / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(f"V-JEPA2 source directory does not exist: {source_root}")
        import src as workspace_src

        source_root_string = str(source_root)
        if source_root_string not in workspace_src.__path__:
            workspace_src.__path__.append(source_root_string)
        return importlib.import_module("src.models.predictor").vit_predictor

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

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_loader = self._import_checkpoint_loader(self._vjepa2_root)
        checkpoint = checkpoint_loader(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("predictor")
        if not isinstance(state_dict, dict):
            raise ValueError(f"checkpoint has no predictor state dict: {checkpoint_path}")
        normalized = _normalize_state_dict(state_dict)
        expected = self.predictor.state_dict()
        compatible = {
            key: value for key, value in normalized.items() if key in expected and expected[key].shape == value.shape
        }
        missing, _ = self.predictor.load_state_dict(compatible, strict=False)
        if "predictor_embed.weight" in missing:
            raise ValueError(f"checkpoint is incompatible with V-JEPA predictor: {checkpoint_path}")
        del checkpoint, state_dict, normalized, compatible, expected

    def _inject(self, z_task_masked: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        u = self.u_projection(U).unsqueeze(1)
        if self.mode == "add":
            return z_task_masked + u
        combined = torch.cat([z_task_masked, u.expand(-1, z_task_masked.size(1), -1)], dim=-1)
        return self.concat_projection(combined)

    def forward(
        self,
        z_task_masked: torch.Tensor,
        U: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        # If confounder branch is disabled, use the masked task representation as is
        if U is None:
            conditioned = z_task_masked
        else:
            conditioned = self._inject(z_task_masked, U)
        return self.predictor(conditioned, [context_indices], [target_indices])