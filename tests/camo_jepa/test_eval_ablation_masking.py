"""CPU tests for the WM-12 masking ablation code. No checkpoint or GPU needed.

Run from the repository root: ``python3 -m pytest tests/camo_jepa -q``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.camo_jepa.evaluation import eval_model_loading as loading
from src.camo_jepa.evaluation.mask_heatmap import cell_to_box, token_to_cell
from src.camo_jepa.evaluation.masking_strategies import (
    MotionWeightedMaskSampler, RandomMaskSampler, SeededBlockSampler, measure_block_mask_fraction)
from src.camo_jepa.evaluation.region_metrics import (
    MeanPredictor, dynamic_mask, patch_flow_magnitude, region_sums, token_cosines)
from src.camo_jepa.perception import MaskSampler

REPO = Path(__file__).resolve().parents[2]
VJEPA2 = REPO / "src" / "vjepa2"
N_TOKENS, PER_T = 15 * 512, 512
needs_vjepa2 = pytest.mark.skipif(not (VJEPA2 / "src" / "masks").is_dir(), reason="src/vjepa2 submodule not initialised")


def _check_partition(context, mask, total=N_TOKENS):
    assert torch.equal(mask, mask.sort().values) and torch.equal(context, context.sort().values)
    assert torch.equal(torch.cat([context, mask]).sort().values, torch.arange(total))


# 1. random and motion-weighted masks: exactly M, no duplicates, context is the complement
@pytest.mark.parametrize("sampler", [RandomMaskSampler(4600, seed=3), MotionWeightedMaskSampler(4600, seed=3)])
def test_samplers_mask_exactly_m_and_partition_tokens(sampler):
    if isinstance(sampler, MotionWeightedMaskSampler):
        sampler.set_scores(torch.rand(1, N_TOKENS, generator=torch.Generator().manual_seed(0)))
    z = torch.randn(1, N_TOKENS, 4)
    z_masked, context, mask = sampler(z)
    assert mask.shape == (1, 4600) and context.shape == (1, N_TOKENS - 4600)
    _check_partition(context[0], mask[0])
    assert torch.equal(z_masked[0], z[0, context[0]])


# 2. motion-weighted masking prefers high flow
def test_motion_sampler_prefers_high_flow_tokens():
    scores = torch.ones(1, N_TOKENS)
    scores[0, : N_TOKENS // 2] = 100.0
    sampler = MotionWeightedMaskSampler(N_TOKENS // 2, seed=0)
    sampler.set_scores(scores)
    mask = sampler(torch.zeros(1, N_TOKENS, 1))[2][0]
    assert (mask < N_TOKENS // 2).float().mean() > 0.9


# 3. random masking ignores flow
def test_random_sampler_is_close_to_uniform():
    mask = RandomMaskSampler(N_TOKENS // 2, seed=0)(torch.zeros(1, N_TOKENS, 1))[2][0]
    assert abs((mask < N_TOKENS // 2).float().mean().item() - 0.5) < 0.05


# 4. dynamic region = top 20% of every transition, ties deterministic
def test_dynamic_mask_takes_103_patches_per_transition_and_breaks_ties_by_index():
    magnitude = torch.rand(N_TOKENS, generator=torch.Generator().manual_seed(1))
    mask = dynamic_mask(magnitude).reshape(15, PER_T)
    assert mask.sum(dim=1).tolist() == [math.ceil(0.2 * PER_T)] * 15 == [103] * 15
    flat = dynamic_mask(torch.zeros(PER_T))
    assert flat[:103].all() and not flat[103:].any()
    assert torch.equal(dynamic_mask(magnitude), dynamic_mask(magnitude.clone()))


# 5. token index <-> (transition, row, col) <-> pixel box on the 1920x1536 frame
def test_token_index_maps_to_cells_and_source_pixels():
    t, row, col = token_to_cell(np.arange(N_TOKENS))
    assert np.array_equal(t * PER_T + row * 32 + col, np.arange(N_TOKENS))
    assert t.max() == 14 and row.max() == 15 and col.max() == 31
    assert tuple(cell_to_box(15, 31, (1920, 1536))) == (1860.0, 1440.0, 1920.0, 1536.0)
    assert tuple(cell_to_box(0, 0, (1920, 1536))) == (0.0, 0.0, 60.0, 96.0)


# 6. per-token cosine and region pooling
def test_token_cosines_and_region_sums():
    a = torch.zeros(3, 8)
    a[:, 0], a[:, 1] = 1.0, -1.0
    b = a.clone()
    b[2] = 0.0
    b[2, 2], b[2, 3] = 1.0, -1.0  # zero-mean and orthogonal to a[2]
    cos = token_cosines(a, b)
    assert torch.allclose(cos["cos"], torch.tensor([1.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(cos["cos_ln"], torch.tensor([1.0, 1.0, 0.0]), atol=1e-5)
    sums = region_sums(torch.tensor([0.2, 0.4, 0.9]), torch.tensor([True, False, True]))
    assert sums["dynamic"] == [pytest.approx(1.1), 2] and sums["static"] == [pytest.approx(0.4), 1]
    assert sums["all"] == [pytest.approx(1.5), 3]


# 7. flow magnitude is averaged per 16x16 patch at full and at 1/8 resolution
def test_patch_flow_magnitude_finds_the_moving_patch():
    full = torch.zeros(2, 2, 256, 512)
    full[1, 0, 80:96, 160:176], full[1, 1, 80:96, 160:176] = 3.0, 4.0
    mag = patch_flow_magnitude(full).reshape(2, PER_T)
    assert mag[1].argmax().item() == 5 * 32 + 10 and mag[1].max().item() == pytest.approx(5.0) and mag[0].max() == 0
    eighth = torch.zeros(1, 2, 32, 64)
    eighth[0, 0, 10:12, 20:22] = 5.0
    mag = patch_flow_magnitude(eighth)
    assert mag.argmax().item() == 5 * 32 + 10 and mag.max().item() == pytest.approx(40.0)  # x8 to input pixels


# 8. load gates refuse partial checkpoints
class _TinyCaMo(nn.Module):
    def __init__(self):
        super().__init__()
        for name in loading.TRAINABLE_MODULES:
            setattr(self, name, nn.Linear(3, 2))


def test_camo_gate_rejects_a_checkpoint_without_one_module(tmp_path):
    source = _TinyCaMo()
    full = {"epoch": 2, **{name: getattr(source, name).state_dict() for name in loading.TRAINABLE_MODULES}}
    torch.save(full, tmp_path / "full.pt")
    report = loading.load_camo_checkpoint(_TinyCaMo(), tmp_path / "full.pt", expect_epoch=2)
    assert report["epoch"] == 2 and set(report["tensors_matched"].values()) == {2}
    torch.save({key: value for key, value in full.items() if key != "fusion"}, tmp_path / "partial.pt")
    with pytest.raises(loading.GateError, match="fusion"):
        loading.load_camo_checkpoint(_TinyCaMo(), tmp_path / "partial.pt")
    with pytest.raises(loading.GateError):
        loading.load_camo_checkpoint(_TinyCaMo(), tmp_path / "full.pt", expect_epoch=5)


def test_flowformer_gate_rejects_a_missing_key_and_a_wrong_value(tmp_path):
    backbone = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    state = {f"module.{key}": value.clone() for key, value in backbone.state_dict().items()}
    torch.save(state, tmp_path / "ok.pth")
    assert loading.check_flowformer_weights(backbone, tmp_path / "ok.pth")["matched"] == 4
    torch.save({key: value for key, value in state.items() if not key.endswith("1.bias")}, tmp_path / "missing.pth")
    with pytest.raises(loading.GateError, match="'missing': 1"):
        loading.check_flowformer_weights(backbone, tmp_path / "missing.pth")
    state["module.0.weight"] = state["module.0.weight"] + 1.0
    torch.save(state, tmp_path / "changed.pth")
    with pytest.raises(loading.GateError, match="'value_mismatch': 1"):
        loading.check_flowformer_weights(backbone, tmp_path / "changed.pth")


# 9. the train-time block sampler, one sample at a time
@needs_vjepa2
def test_block_sampler_partitions_and_its_mask_fraction_is_reproducible():
    sampler = MaskSampler(0.7, image_size=(512, 256), vjepa2_root=VJEPA2)
    _, context, mask = SeededBlockSampler(sampler, seed=0)(torch.zeros(1, N_TOKENS, 1))
    _check_partition(context[0], mask[0])
    first = measure_block_mask_fraction(sampler, 15, n_draws=50, seed=7)
    again = measure_block_mask_fraction(sampler, 15, n_draws=50, seed=7)
    assert 0.0 < first["mean"] < 1.0 and first == again
    with pytest.raises(ValueError):
        SeededBlockSampler(sampler)(torch.zeros(2, N_TOKENS, 1))


# 10. Mean-Predictor baseline
def test_mean_predictor_matches_hand_computation():
    tokens = torch.randn(3, 4, 6, generator=torch.Generator().manual_seed(2))
    means = MeanPredictor(4, 6)
    means.update(tokens)
    positions = torch.tensor([1, 3])
    target = torch.randn(2, 6, generator=torch.Generator().manual_seed(3))
    expected = torch.nn.functional.cosine_similarity(tokens.mean(0)[positions], target, dim=-1)
    got = token_cosines(means.predict(positions, "cos", "position"), target)["cos"]
    assert torch.allclose(got, expected, atol=1e-6)
    global_mean = tokens.reshape(-1, 6).mean(0)
    assert torch.allclose(means.predict(positions, "cos", "global")[0], global_mean, atol=1e-6)
