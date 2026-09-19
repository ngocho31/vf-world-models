"""Dynamic/static regions and per-token scores for the masking ablation (WM-12).

Token ``i`` of the flattened window sequence ``[(T-1) * N_p]`` belongs to
transition ``t = i // N_p`` (clip ``[t, t+1]``) and patch ``p = i % N_p``, laid
out row-major on the ``16 x 32`` grid of the 512x256 input
(``pipeline/phase1.py:163-165``, ``perception/static_encoder.py:95``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PATCH_GRID = (16, 32)  # (rows, cols) for image_size (512, 256) and 16x16 patches
REGIONS = ("dynamic", "static", "all")


def patch_flow_magnitude(
    flow: torch.Tensor, grid: tuple[int, int] = PATCH_GRID, input_width: int = 512
) -> torch.Tensor:
    """Mean ``|flow|`` over each patch: ``[T-1, 2, h, w]`` -> ``[(T-1) * rows * cols]``.

    Works for full-resolution flow (256x512) and for the 1/8-resolution flow
    (32x64) FlowFormer++ returns in eval mode: the flow grid only has to tile
    the patch grid. Values are rescaled to pixels of the ``input_width``-wide
    model input so runs at different flow resolutions report the same unit.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must have shape [T-1, 2, h, w], got {tuple(flow.shape)}")
    rows, cols = grid
    height, width = flow.shape[-2:]
    if height % rows or width % cols:
        raise ValueError(f"flow resolution {(height, width)} does not tile the patch grid {grid}")
    magnitude = torch.linalg.vector_norm(flow.float(), dim=1, keepdim=True)
    pooled = F.avg_pool2d(magnitude, kernel_size=(height // rows, width // cols))
    return pooled.flatten() * (input_width / width)


def dynamic_mask(
    magnitude: torch.Tensor, tokens_per_transition: int = PATCH_GRID[0] * PATCH_GRID[1],
    top_fraction: float = 0.2,
) -> torch.Tensor:
    """True for the top ``top_fraction`` patches of every transition by flow magnitude.

    ``ceil(top_fraction * N_p)`` patches per transition (103 of 512 at 20%);
    ties go to the lower patch index, so the split is deterministic.
    """
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be in (0, 1)")
    per_transition = magnitude.reshape(-1, tokens_per_transition)
    k = math.ceil(top_fraction * tokens_per_transition)
    order = torch.argsort(per_transition, dim=1, descending=True, stable=True)
    mask = torch.zeros_like(per_transition, dtype=torch.bool)
    mask.scatter_(1, order[:, :k], True)
    return mask.flatten()


def token_cosines(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-token cosine ``[..., D] -> [...]``, plain and after per-token layer norm.

    ``cos_ln`` compares the two vectors after standardising each over its
    channels (the Pearson correlation of the channels). That is the geometry of
    the training loss, which layer-norms prediction and target before the
    Smooth L1 (``evaluation/losses.py:8-12``), so a per-token offset or scale the
    loss never constrained cannot move it. ``cos`` is ``evaluation/metrics.py``'s
    plain cosine.
    """
    pred, target = pred.float(), target.float()
    dim = pred.shape[-1]
    return {
        "cos_ln": F.cosine_similarity(F.layer_norm(pred, (dim,)), F.layer_norm(target, (dim,)), dim=-1),
        "cos": F.cosine_similarity(pred, target, dim=-1),
    }


def region_sums(values: torch.Tensor, is_dynamic: torch.Tensor) -> dict[str, list[float]]:
    """``{region: [sum, count]}`` of per-token ``values`` split by ``is_dynamic``."""
    values = values.double().flatten()
    is_dynamic = is_dynamic.flatten().bool()
    out = {}
    for region, keep in (("dynamic", is_dynamic), ("static", ~is_dynamic), ("all", torch.ones_like(is_dynamic))):
        out[region] = [float(values[keep].sum()), int(keep.sum())]
    return out


class MeanPredictor:
    """The Mean-Predictor baseline: predict ``z*`` by its average over the evaluated windows.

    Pass 1 feeds every window's ``z*`` tokens ``[T-1, N_p, D]`` to ``update``;
    pass 2 asks for the prediction at the masked tokens' patch positions. Means
    are kept both of the raw vectors (scored with ``cos``) and of the
    layer-normed vectors (scored with ``cos_ln``). They include the tokens being
    scored, which can only help the baseline.
    """

    def __init__(self, tokens_per_transition: int, dim: int, device: torch.device | str = "cpu") -> None:
        shape = (tokens_per_transition, dim)
        self.sums = {
            "cos": torch.zeros(shape, dtype=torch.float64, device=device),
            "cos_ln": torch.zeros(shape, dtype=torch.float64, device=device),
        }
        self.transitions = 0

    def update(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != self.sums["cos"].shape:
            raise ValueError(f"expected tokens [T-1, {self.sums['cos'].shape[0]}, D], got {tuple(tokens.shape)}")
        tokens = tokens.float()
        self.sums["cos"] += tokens.double().sum(dim=0)
        self.sums["cos_ln"] += F.layer_norm(tokens, (tokens.shape[-1],)).double().sum(dim=0)
        self.transitions += tokens.shape[0]

    def predict(self, positions: torch.Tensor, metric: str, scope: str) -> torch.Tensor:
        """Mean vectors for patch ``positions`` ``[M]``: per position or one global mean."""
        if self.transitions == 0:
            raise RuntimeError("MeanPredictor has no data; run the first pass")
        total = self.sums[metric]
        if scope == "position":
            return (total / self.transitions)[positions.to(total.device)].float()
        if scope == "global":
            mean = total.sum(dim=0) / (self.transitions * total.shape[0])
            return mean.float().expand(positions.numel(), -1)
        raise ValueError(f"unknown scope {scope!r}")


def score_masked_tokens(
    z_pred: torch.Tensor, z_star: torch.Tensor, mask_indices: torch.Tensor,
    is_dynamic: torch.Tensor, mean_predictor: MeanPredictor,
    tokens_per_transition: int = PATCH_GRID[0] * PATCH_GRID[1],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """``{predictor: {metric: {region: [sum, count]}}}`` over the masked tokens of one window.

    ``z_pred`` and ``z_star`` are ``[M, D]`` in the order of ``mask_indices``
    ``[M]``; ``is_dynamic`` is the window's ``[N]`` region flag.
    """
    positions = mask_indices % tokens_per_transition
    dynamic = is_dynamic.to(mask_indices.device)[mask_indices]
    scores = {"model": token_cosines(z_pred, z_star)}
    for scope in ("position", "global"):
        scores[f"mean_{scope}"] = {
            metric: token_cosines(mean_predictor.predict(positions, metric, scope).to(z_star.device), z_star)[metric]
            for metric in ("cos_ln", "cos")
        }
    return {
        predictor: {metric: region_sums(values, dynamic) for metric, values in by_metric.items()}
        for predictor, by_metric in scores.items()
    }
