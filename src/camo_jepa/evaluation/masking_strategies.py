"""Mask samplers for the masking ablation (WM-12).

Each sampler follows the ``MaskSampler.forward`` contract that
``CaMoJEPAPipeline.forward`` calls at ``pipeline/phase1.py:168``::

    forward(z_task [B, N, D]) -> (z_task_masked [B, K, D], context_indices [B, K], mask_indices [B, M])

so it can replace ``pipeline.mask_sampler`` without touching model code.
Mask and context indices are sorted ascending and are exact complements.

* ``block``  - the train-time V-JEPA2 multiblock sampler, one sample per call.
* ``random`` - ``M`` tokens drawn uniformly without replacement.
* ``motion`` - ``M`` tokens drawn without replacement with probability
  proportional to the token's optical-flow magnitude (plus ``eps``).
"""

from __future__ import annotations

import torch
from torch import nn

from ..perception.masking import MaskSampler


def split_by_mask(
    z_task: torch.Tensor, mask_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(z_task_masked, context_indices, mask_indices)`` for one mask per row."""
    total, dim = z_task.shape[1], z_task.shape[2]
    mask_indices = mask_indices.to(device=z_task.device, dtype=torch.long).sort(dim=1).values
    if (mask_indices[:, 1:] == mask_indices[:, :-1]).any():
        raise ValueError("mask_indices contains duplicates")
    if mask_indices.numel() and (mask_indices.min() < 0 or mask_indices.max() >= total):
        raise ValueError(f"mask_indices must lie in [0, {total})")
    context_indices = MaskSampler.complement_indices(mask_indices, total)
    z_task_masked = torch.gather(z_task, 1, context_indices.unsqueeze(-1).expand(-1, -1, dim))
    return z_task_masked, context_indices, mask_indices


class RandomMaskSampler(nn.Module):
    """Mask ``num_masked`` tokens chosen uniformly at random, independently per row."""

    def __init__(self, num_masked: int, seed: int = 0) -> None:
        super().__init__()
        if num_masked < 1:
            raise ValueError("num_masked must be positive")
        self.num_masked = num_masked
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, z_task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, total, _ = z_task.shape
        if self.num_masked >= total:
            raise ValueError(f"cannot mask {self.num_masked} of {total} tokens and keep a context")
        rows = [torch.randperm(total, generator=self.generator)[: self.num_masked] for _ in range(batch)]
        return split_by_mask(z_task, torch.stack(rows))


class MotionWeightedMaskSampler(nn.Module):
    """Mask ``num_masked`` tokens, each draw proportional to flow magnitude + ``eps``.

    Call ``set_scores`` with the per-token magnitude ``[B, N]`` of the current
    window before the pipeline forward. ``eps`` keeps zero-flow tokens drawable
    once the moving ones are used up, so every row gets exactly ``num_masked``.
    """

    def __init__(self, num_masked: int, seed: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        if num_masked < 1:
            raise ValueError("num_masked must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_masked = num_masked
        self.eps = eps
        self.generator = torch.Generator().manual_seed(seed)
        self.scores: torch.Tensor | None = None

    def set_scores(self, scores: torch.Tensor) -> None:
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, tokens]")
        if not torch.isfinite(scores).all():
            raise ValueError("scores must be finite")
        self.scores = scores.detach().to(device="cpu", dtype=torch.float64)

    def forward(self, z_task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scores is None:
            raise RuntimeError("call set_scores() with the window's flow magnitude first")
        batch, total, _ = z_task.shape
        if tuple(self.scores.shape) != (batch, total):
            raise ValueError(f"scores shape {tuple(self.scores.shape)} does not match tokens {(batch, total)}")
        if self.num_masked >= total:
            raise ValueError(f"cannot mask {self.num_masked} of {total} tokens and keep a context")
        weights = self.scores.clamp_min(0.0) + self.eps
        mask = torch.multinomial(weights, self.num_masked, replacement=False, generator=self.generator)
        return split_by_mask(z_task, mask)


class SeededBlockSampler(nn.Module):
    """The train-time multiblock sampler (``perception/masking.py``), made reproducible.

    ``_MaskGenerator`` draws the block size from its own call counter but places
    the blocks with the global torch RNG (``multiseq_multiblock3d.py:173-177``);
    this wrapper reseeds that RNG on every call. It accepts one sample at a time
    because the generator truncates every row of a batch to the shortest mask in
    it (``multiseq_multiblock3d.py:226-235``).
    """

    def __init__(self, sampler: MaskSampler, seed: int = 0) -> None:
        super().__init__()
        self.sampler = sampler
        self.seed = seed
        self.calls = 0

    def forward(self, z_task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_task.shape[0] != 1:
            raise ValueError("SeededBlockSampler samples one window at a time (batch size 1)")
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.seed + self.calls)
            self.calls += 1
            return self.sampler(z_task)


def clone_mask_sampler(sampler: MaskSampler) -> MaskSampler:
    """A fresh ``MaskSampler`` with the same settings and a reset call counter."""
    return MaskSampler(
        sampler.mask_ratio,
        image_size=sampler.image_size,
        patch_size=sampler.patch_size,
        spatial_pred_scale=sampler.spatial_pred_scale,
        aspect_ratio=sampler.aspect_ratio,
        npred=sampler.npred,
        vjepa2_root=sampler.vjepa2_root,
    )


def measure_block_mask_fraction(
    sampler: MaskSampler, num_transitions: int, n_draws: int = 400, seed: int = 0
) -> dict[str, float]:
    """Fraction of tokens the multiblock sampler masks per sample, over ``n_draws`` draws.

    Runs on a clone so the pipeline's own sampler keeps its counter. The mean
    sets ``M`` for the random and motion-weighted samplers, which is what makes
    the three strategies mask the same number of tokens on average
    (``mask_ratio`` is the temporal length of a block, not a token fraction).
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    width, height = sampler.image_size
    total = num_transitions * (height // sampler.patch_size) * (width // sampler.patch_size)
    block = SeededBlockSampler(clone_mask_sampler(sampler), seed=seed)
    probe = torch.zeros(1, total, 1)
    fractions = torch.tensor(
        [block(probe)[2].shape[1] / total for _ in range(n_draws)], dtype=torch.float64
    )
    return {
        "mean": float(fractions.mean()),
        "std": float(fractions.std(unbiased=True)) if n_draws > 1 else 0.0,
        "min": float(fractions.min()),
        "max": float(fractions.max()),
        "n_draws": n_draws,
        "tokens_per_window": total,
        "seed": seed,
    }
