"""Build a CaMo-JEPA pipeline for evaluation and prove every stage holds its weights.

The model code loads weights leniently: the ViT and FlowFormer++ loaders keep
only keys that match by name and shape (``perception/static_encoder.py:77-85``,
``perception/motion_encoder.py:105-117``), and ``load_checkpoint`` skips any
trainable module absent from the file (``pipeline/checkpoints.py:69-73``). A
partial load raises nothing. The gates below re-read each checkpoint and
compare it tensor by tensor with what the pipeline holds, so scores can only be
computed on a model whose weights are the checkpoint's. Shared by the WM-11 and
WM-12 evaluations.
"""

from __future__ import annotations

import os
import pickle
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from ..config import CaMoJEPAConfig
from ..perception.motion_encoder import _normalize_state_dict as normalize_flowformer_keys
from ..perception.static_encoder import _normalize_state_dict as normalize_vit_keys
from ..pipeline import CaMoJEPAPipeline, load_checkpoint

TRAINABLE_MODULES = ("flow_encoder", "fusion", "factorizer", "confounder", "predictor")
# Encoder keys that may legitimately be absent from the Drive-JEPA checkpoint.
# Empty: every ViT-L tensor must come from the file. Add a key only with a reason.
VIT_ALLOWED_MISSING: tuple[str, ...] = ()
# Mean token L2 norm on a real window. The trained Drive-JEPA encoders give ~105
# (104.9-105.8 measured in earlier project runs); an untrained ViT ends in a
# unit-gain LayerNorm and gives sqrt(1024) = 32. The bar sits between the two.
MIN_TOKEN_L2 = 60.0


class GateError(RuntimeError):
    """A load gate failed; ``report`` holds what was measured."""

    def __init__(self, gate: str, report: dict) -> None:
        super().__init__(f"load gate '{gate}' failed: {report}")
        self.gate = gate
        self.report = report


@contextmanager
def trusted_full_unpickling():
    """Let ``torch.load`` unpickle non-tensor entries while the pipeline is built.

    The Drive-JEPA checkpoint holds non-tensor entries that torch>=2.6 refuses
    under its default ``weights_only=True``, and ``robust_checkpoint_loader``
    does not pass the flag. The variable only affects calls that leave
    ``weights_only`` unset and is restored on exit. Known checkpoints only.
    """
    name = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def build_pipeline(config: CaMoJEPAConfig, device: torch.device | str) -> CaMoJEPAPipeline:
    with trusted_full_unpickling():
        pipeline = CaMoJEPAPipeline(config)
    return pipeline.to(device).eval()


def _torch_load(path: str | Path, weights_only: bool, mmap: bool) -> object:
    if mmap:
        try:  # memory-mapping needs the zip format; fall back to a plain read
            return torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
        except RuntimeError:
            pass
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def load_file(path: str | Path, weights_only_first: bool = True, mmap: bool = False) -> tuple[object, bool]:
    """``torch.load`` on CPU. Returns ``(object, weights_only_used)``.

    Tries ``weights_only=True`` first and falls back to full unpickling only
    when the file needs it; use the fallback for known checkpoints only.
    """
    if weights_only_first:
        try:
            return _torch_load(path, True, mmap), True
        except pickle.UnpicklingError:
            pass
    return _torch_load(path, False, mmap), False


def compare_state(module_state: dict, reference: dict, gate: str, allow_missing: tuple[str, ...] = ()) -> dict:
    """Every tensor of ``module_state`` must exist in ``reference`` with equal shape and value."""
    expected, provided = set(module_state), set(reference)
    missing = sorted(expected - provided - set(allow_missing))
    common = sorted(expected & provided)
    shape_clash = [k for k in common if tuple(module_state[k].shape) != tuple(reference[k].shape)]
    clash = set(shape_clash)
    value_mismatch = [
        k for k in common if k not in clash
        and not torch.equal(module_state[k].detach().cpu(), reference[k].detach().cpu().to(module_state[k].dtype))
    ]
    report = {
        "expected": len(expected),
        "matched": len(common) - len(shape_clash) - len(value_mismatch),
        "missing": len(missing),
        "shape_clash": len(shape_clash),
        "value_mismatch": len(value_mismatch),
        "unexpected_in_checkpoint": len(provided - expected),
        "allowed_missing": sorted(set(allow_missing) & (expected - provided)),
        "examples": {
            "missing": missing[:5], "shape_clash": shape_clash[:5],
            "value_mismatch": value_mismatch[:5], "unexpected": sorted(provided - expected)[:5],
        },
    }
    if missing or shape_clash or value_mismatch:
        raise GateError(gate, report)
    return report


def load_camo_checkpoint(pipeline: nn.Module, path: str | Path, expect_epoch: int | None = None) -> dict:
    """Gate 1: all five trainable modules present, loaded strictly, values equal to the file."""
    checkpoint, weights_only = load_file(path)
    absent = [name for name in TRAINABLE_MODULES if name not in checkpoint]
    if absent:
        raise GateError("camo", {"absent_modules": absent, "keys": sorted(map(str, checkpoint))})
    try:
        if weights_only:
            _, epoch, _ = load_checkpoint(path, pipeline, strict=True)
        else:  # load_checkpoint's bare torch.load would refuse this file on torch>=2.6
            for name in TRAINABLE_MODULES:
                getattr(pipeline, name).load_state_dict(checkpoint[name], strict=True)
            epoch = checkpoint.get("epoch")
    except RuntimeError as error:  # strict mismatch, e.g. a checkpoint from another config
        raise GateError("camo.strict_load", {"error": str(error)[:2000]}) from error
    per_module = {
        name: compare_state(getattr(pipeline, name).state_dict(), checkpoint[name], f"camo.{name}")["matched"]
        for name in TRAINABLE_MODULES
    }
    report = {"modules": list(TRAINABLE_MODULES), "tensors_matched": per_module, "epoch": epoch,
              "expected_epoch": expect_epoch, "weights_only_load": weights_only}
    if expect_epoch is not None and epoch != expect_epoch:
        raise GateError("camo.epoch", report)
    return report


def check_flowformer_weights(backbone: nn.Module, path: str | Path) -> dict:
    """Gate 2: the FlowFormer++ backbone holds every tensor of the checkpoint file."""
    state, _ = load_file(path)
    if isinstance(state, dict):  # same unwrapping as perception/motion_encoder.py:99-103
        for key in ("model", "state_dict", "flow_encoder", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    reference = normalize_flowformer_keys(state)
    report = compare_state(backbone.state_dict(), reference, "flowformer")
    report["checkpoint_tensors"] = len(reference)
    return report


def check_vit_weights(pipeline: nn.Module, path: str | Path) -> dict:
    """Gate 3a: context <- ``encoder`` and target <- ``target_encoder`` of the Drive-JEPA file."""
    checkpoint, _ = load_file(path, weights_only_first=False, mmap=True)
    report = {}
    for attribute, key in (("context_encoder", "encoder"), ("target_encoder", "target_encoder")):
        if key not in checkpoint:
            raise GateError(f"vit.{attribute}", {"absent_key": key, "keys": sorted(map(str, checkpoint))})
        reference = normalize_vit_keys(checkpoint[key])
        state = getattr(pipeline, attribute).backbone.state_dict()
        report[attribute] = {"checkpoint_key": key, **compare_state(state, reference, f"vit.{attribute}", VIT_ALLOWED_MISSING)}
    del checkpoint
    return report


def check_token_norms(context_tokens: torch.Tensor, target_tokens: torch.Tensor, minimum: float = MIN_TOKEN_L2) -> dict:
    """Gate 3b: trained encoders give token norms far above an untrained ViT's sqrt(D)."""
    report = {
        "context_mean_l2": float(context_tokens.float().norm(dim=-1).mean()),
        "target_mean_l2": float(target_tokens.float().norm(dim=-1).mean()),
        "untrained_reference_l2": float(context_tokens.shape[-1]) ** 0.5,
        "minimum": minimum,
    }
    if min(report["context_mean_l2"], report["target_mean_l2"]) <= minimum:
        raise GateError("vit.token_l2", report)
    return report
