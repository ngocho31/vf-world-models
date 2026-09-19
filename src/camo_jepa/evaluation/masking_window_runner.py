"""Per-window passes of the masking ablation (WM-12).

Pass 1 runs the two ViT-L encoders over every window for the Mean-Predictor.
In pass 2 the frozen stages (both encoders and FlowFormer++) run once per
window, and the pipeline's own ``forward`` (``pipeline/phase1.py:128-193``)
then runs once per mask strategy with those stages answered from a cache, so
everything after them (flow encoder, fusion, factorizer, confounder, mask
sampler, predictor) is the unmodified model code.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from ..contracts import FrameBatch
from .region_metrics import MeanPredictor, dynamic_mask, patch_flow_magnitude, region_sums, score_masked_tokens, token_cosines
from .window_bootstrap import flatten_scores

Z_STAR_SOURCES = ("target_encoder", "context_encoder")  # primary, secondary


class _FrozenStageCache(nn.Module):
    """Stands in for a frozen stage and returns its precomputed output for one window."""

    def __init__(self, images: torch.Tensor, output: torch.Tensor) -> None:
        super().__init__()
        self._images = images
        self._output = output

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images is not self._images:
            raise RuntimeError("cached frozen stage called with another window's images")
        return self._output

    forward_tokens = forward


@contextmanager
def cached_frozen_stages(pipeline: nn.Module, images, context_tokens, target_tokens, flow):
    names = ("context_encoder", "target_encoder", "flow_estimator")
    originals = {name: getattr(pipeline, name) for name in names}
    outputs = dict(zip(names, (context_tokens, target_tokens, flow)))
    try:
        for name in names:
            setattr(pipeline, name, _FrozenStageCache(images, outputs[name]))
        yield
    finally:
        for name, module in originals.items():
            setattr(pipeline, name, module)


@contextmanager
def mask_sampler_as(pipeline: nn.Module, sampler: nn.Module):
    original = pipeline.mask_sampler
    pipeline.mask_sampler = sampler
    try:
        yield
    finally:
        pipeline.mask_sampler = original


def compute_flow(flow_estimator: nn.Module, images: torch.Tensor, chunk: int) -> torch.Tensor:
    """Flow for every consecutive pair, ``chunk`` pairs per call; pairs are independent."""
    pairs = images.shape[1] - 1
    parts = [flow_estimator(images[:, start:min(start + chunk, pairs) + 1]) for start in range(0, pairs, chunk)]
    return torch.cat(parts, dim=1)


def window_frame(dataset, window_id: int, offset: int) -> Path:
    """Source image of frame ``offset`` of a window (read-only use of the loader's window list)."""
    window = dataset._windows[window_id]
    paths = dataset._load_episode(window.episode_path)["image_paths"]
    return window.dataset_root / str(paths[window.start_index + offset])


def to_device(batch: FrameBatch, device: torch.device) -> FrameBatch:
    ego = batch.ego_motion.to(device) if batch.ego_motion is not None else None
    return FrameBatch(images=batch.images.to(device), ego_motion=ego)


@torch.no_grad()
def accumulate_means(pipeline: nn.Module, loader, device: torch.device, on_first_window=None) -> dict[str, MeanPredictor]:
    """Pass 1: per-position means of both z* sources over the evaluated windows."""
    means: dict[str, MeanPredictor] = {}
    for step, batch in enumerate(loader):
        images = batch.images.to(device)
        tokens = {
            "context_encoder": pipeline.context_encoder.forward_tokens(images),
            "target_encoder": pipeline.target_encoder.forward_tokens(images),
        }
        if step == 0 and on_first_window is not None:
            on_first_window(tokens["context_encoder"], tokens["target_encoder"])
        for source in Z_STAR_SOURCES:
            if source not in means:
                means[source] = MeanPredictor(tokens[source].shape[2], tokens[source].shape[3], device)
            means[source].update(tokens[source][0])
    return means


def _check_partition(context: torch.Tensor, mask: torch.Tensor, total: int) -> None:
    joined = torch.cat([context, mask]).sort().values
    if joined.numel() != total or not torch.equal(joined, torch.arange(total, device=joined.device)):
        raise RuntimeError("context and mask indices do not partition the window's tokens")


@torch.no_grad()
def score_window(pipeline: nn.Module, batch: FrameBatch, samplers: dict, means: dict[str, MeanPredictor],
                 flow_chunk: int, top_fraction: float, keep_tokens: bool = False) -> tuple[dict, dict]:
    """Pass 2 for one window: ``(row, extras)``.

    ``row`` maps ``predictor/mask/target/metric/region`` (plus ``masked_dynamic/<mask>``
    and ``flow_px/<region>``) to ``[sum, count]``. ``extras`` holds measured shapes
    and, with ``keep_tokens``, the per-token data the heatmaps need.
    """
    images = batch.images
    context_tokens = pipeline.context_encoder.forward_tokens(images)
    target_tokens = pipeline.target_encoder.forward_tokens(images)
    flow = compute_flow(pipeline.flow_estimator, images, flow_chunk)
    per_transition = context_tokens.shape[2]
    magnitude = patch_flow_magnitude(flow[0], input_width=images.shape[-1])
    dynamic = dynamic_mask(magnitude, tokens_per_transition=per_transition, top_fraction=top_fraction)
    for sampler in samplers.values():
        if hasattr(sampler, "set_scores"):  # motion-weighted masking reads this window's flow
            sampler.set_scores(magnitude.unsqueeze(0))
    z_star = {"target_encoder": target_tokens.flatten(1, 2)[0], "context_encoder": context_tokens.flatten(1, 2)[0]}
    total = z_star["target_encoder"].shape[0]

    row = {f"flow_px/{region}": pair for region, pair in region_sums(magnitude, dynamic).items()}
    extras = {"flow_shape": list(flow.shape), "tokens_per_window": total,
              "transition_flow_px": magnitude.reshape(-1, per_transition).mean(dim=1).tolist()}
    if keep_tokens:
        extras["tokens"] = {"dynamic": dynamic.cpu().numpy(), "magnitude": magnitude.cpu().numpy(), "masks": {}}
    with cached_frozen_stages(pipeline, images, context_tokens, target_tokens, flow):
        for name, sampler in samplers.items():
            with mask_sampler_as(pipeline, sampler):
                out = pipeline(batch)
            mask = out.mask_indices[0]
            _check_partition(out.context_indices[0], mask, total)
            if getattr(sampler, "num_masked", mask.numel()) != mask.numel():
                raise RuntimeError(f"{name} masked {mask.numel()} tokens, expected {sampler.num_masked}")
            if not torch.equal(out.target_patches[0], z_star["target_encoder"][mask]):
                raise RuntimeError("ModelOutput.target_patches is not target_encoder at mask_indices")
            row[f"masked_dynamic/{name}"] = [float(dynamic[mask].sum()), int(mask.numel())]
            for source in Z_STAR_SOURCES:
                scores = score_masked_tokens(out.z_pred[0], z_star[source][mask], mask, dynamic, means[source], per_transition)
                for predictor, by_metric in scores.items():
                    row.update(flatten_scores(f"{predictor}/{name}/{source}", by_metric))
            if keep_tokens:
                cos_ln = token_cosines(out.z_pred[0], z_star["target_encoder"][mask])["cos_ln"]
                extras["tokens"]["masks"][name] = (mask.cpu().numpy(), cos_ln.cpu().numpy())
    dynamic_features = out.dynamic_features
    extras["motion_tokens_per_transition"] = None if dynamic_features is None else int(dynamic_features.shape[2])
    return row, extras
