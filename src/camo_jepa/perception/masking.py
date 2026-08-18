"""Spatiotemporal block mask sampling using V-JEPA2 ``_MaskGenerator``.

Loads ``_MaskGenerator`` directly from
``vf-drive-jepa/vjepa2/src/masks/multiseq_multiblock3d.py`` and wraps it so
that ``MaskSampler.forward`` accepts the flattened ``z_task`` patch token tensor
``[B, (T-1)*N_patches, D]`` produced by the CaMo-JEPA pipeline.

Configuration passed to ``_MaskGenerator``:
  * ``crop_size``               = ``(height, width)``  from ``image_size``
  * ``spatial_patch_size``      = ``(patch_size, patch_size)``
    → ``H_p = height // patch_size``,  ``W_p = width // patch_size``
  * ``num_frames``              = ``T-1``  (number of transitions in a window)
  * ``temporal_patch_size``     = 1  (each transition is already one token-step)
  * ``temporal_pred_mask_scale``= ``(mask_ratio, mask_ratio)``
  * ``spatial_pred_mask_scale`` = ``spatial_pred_scale``
  * ``aspect_ratio`` / ``npred`` passed through directly
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch import nn


class MaskSampler(nn.Module):
    """Wrap V-JEPA2 ``_MaskGenerator`` for spatiotemporal block masking.

    Args:
        mask_ratio: Fraction of temporal transitions to mask (target block).
        image_size: ``(width, height)`` in pixels.
        patch_size: ViT patch size (default 16).
        spatial_pred_scale: Spatial area fraction range for the block.
        aspect_ratio: Spatial aspect-ratio range for the block.
        npred: Number of independent block samples whose union is the target.
        vjepa2_root: Path to the V-JEPA2 repo (``vf-drive-jepa/vjepa2``).
            Required for loading ``_MaskGenerator``.
    """

    def __init__(
        self,
        mask_ratio: float,
        image_size: tuple[int, int] = (512, 256),
        patch_size: int = 16,
        spatial_pred_scale: tuple[float, float] = (0.2, 0.8),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        npred: int = 4,
        vjepa2_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        self.mask_ratio = mask_ratio
        self.image_size = image_size          # (width, height)
        self.patch_size = patch_size
        self.spatial_pred_scale = spatial_pred_scale
        self.aspect_ratio = aspect_ratio
        self.npred = npred
        self.vjepa2_root = (
            Path(vjepa2_root).expanduser().resolve() if vjepa2_root is not None else None
        )
        self._MaskGeneratorCls = self._import_mask_generator()
        # Cache generators keyed by T_p (number of transitions)
        self._generator_cache: dict[int, object] = {}

    # ------------------------------------------------------------------
    # Load V-JEPA2 _MaskGenerator
    # ------------------------------------------------------------------

    def _import_mask_generator(self):
        """Import ``_MaskGenerator`` from the V-JEPA2 source tree."""
        if self.vjepa2_root is None:
            raise ValueError(
                "vjepa2_root must be set to load _MaskGenerator from "
                "vf-drive-jepa/vjepa2/src/masks/multiseq_multiblock3d.py"
            )
        source_root = self.vjepa2_root / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"V-JEPA2 source directory does not exist: {source_root}"
            )
        import src as workspace_src

        src_str = str(source_root)
        if src_str not in workspace_src.__path__:
            workspace_src.__path__.append(src_str)

        module = importlib.import_module("src.masks.multiseq_multiblock3d")
        cls = getattr(module, "_MaskGenerator", None)
        if cls is None:
            raise AttributeError(
                "_MaskGenerator not found in src.masks.multiseq_multiblock3d"
            )
        return cls

    def _get_generator(self, T_p: int):
        """Return (and cache) a ``_MaskGenerator`` configured for ``T_p`` transitions."""
        gen = self._generator_cache.get(T_p)
        if gen is not None:
            return gen
        width, height = self.image_size
        gen = self._MaskGeneratorCls(
            crop_size=(height, width),              # (H, W) in pixels
            num_frames=T_p,                         # one token-step per transition
            spatial_patch_size=(self.patch_size, self.patch_size),
            temporal_patch_size=1,                  # already transition-level
            spatial_pred_mask_scale=self.spatial_pred_scale,
            temporal_pred_mask_scale=(self.mask_ratio, self.mask_ratio),
            aspect_ratio=self.aspect_ratio,
            npred=self.npred,
            max_context_frames_ratio=1.0,
            max_keep=None,
            inv_block=False,
            full_complement=False,
            pred_full_complement=False,
        )
        self._generator_cache[T_p] = gen
        return gen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, z_task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply spatiotemporal block masking.

        Args:
            z_task: ``[B, (T-1)*N_patches, D]`` — flattened patch tokens.

        Returns:
            z_task_masked:   ``[B, K, D]`` — unmasked context tokens.
            context_indices: ``[B, K]``    — indices of context tokens.
            mask_indices:    ``[B, M]``    — indices of masked target tokens.
        """
        if z_task.ndim != 3:
            raise ValueError("z_task must have shape [batch, tokens, latent_dim]")
        B, total_tokens, D = z_task.shape

        width, height = self.image_size
        N_patches = (height // self.patch_size) * (width // self.patch_size)
        if total_tokens % N_patches != 0:
            raise ValueError(
                f"z_task token count {total_tokens} is not divisible by "
                f"N_patches={N_patches} (image_size={self.image_size}, "
                f"patch_size={self.patch_size})."
            )
        T_p = total_tokens // N_patches

        generator = self._get_generator(T_p)
        # _MaskGenerator.__call__ returns (context_indices [B,K], target_indices [B,M])
        context_indices, mask_indices = generator(B)

        context_indices = context_indices.to(device=z_task.device, dtype=torch.long)
        mask_indices    = mask_indices.to(device=z_task.device, dtype=torch.long)

        z_task_masked = torch.gather(
            z_task,
            1,
            context_indices.unsqueeze(-1).expand(-1, -1, D),
        )
        return z_task_masked, context_indices, mask_indices

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def complement_indices(mask_indices: torch.Tensor, token_count: int) -> torch.Tensor:
        """Build context indices as complement of ``mask_indices``."""
        if mask_indices.ndim != 2:
            raise ValueError("mask_indices must have shape [batch, masked_count]")
        all_idx = torch.arange(token_count, device=mask_indices.device).unsqueeze(0).expand(
            mask_indices.size(0), -1
        )
        keep = torch.ones_like(all_idx, dtype=torch.bool)
        keep.scatter_(1, mask_indices, False)
        return all_idx[keep].reshape(mask_indices.size(0), token_count - mask_indices.size(1))
