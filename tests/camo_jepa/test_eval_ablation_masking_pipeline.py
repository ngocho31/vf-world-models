"""CPU integration tests for the WM-12 window runner, report and heatmap.

``_FakePipeline`` borrows the real ``CaMoJEPAPipeline.forward`` and the real
flow encoder, fusion, factorizer, confounder and mask sampler; only the frozen
encoders, FlowFormer++ and the predictor are cheap stand-ins, so no checkpoint
is needed. The stand-in flow has the 1/8 resolution FlowFormer++ returns in eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.camo_jepa.causal import ConfounderGRU, LatentFactorizer
from src.camo_jepa.config import CaMoJEPAConfig
from src.camo_jepa.contracts import FrameBatch
from src.camo_jepa.data import CaMoEpisodeDataset
from src.camo_jepa.evaluation.eval_environment import IdentityError, _arch_runs_on, sha256_file, verify_files
from src.camo_jepa.evaluation.mask_heatmap import render_mask_heatmap
from src.camo_jepa.evaluation.masking_report import MASKS, METRICS, TARGETS, build_report, headline_lines, summarize
from src.camo_jepa.evaluation.masking_strategies import MotionWeightedMaskSampler, RandomMaskSampler, SeededBlockSampler
from src.camo_jepa.evaluation.masking_window_runner import accumulate_means, compute_flow, score_window
from src.camo_jepa.evaluation.region_metrics import REGIONS, dynamic_mask
from src.camo_jepa.evaluation.window_bootstrap import (
    WindowSeries, bootstrap_indices, contrast_summary, paired_summary, ratio_summary)
from src.camo_jepa.perception import FlowTokenEncoder, GatedCrossAttentionFusion, MaskSampler
from src.camo_jepa.pipeline import CaMoJEPAPipeline

VJEPA2 = Path(__file__).resolve().parents[2] / "src" / "vjepa2"
needs_vjepa2 = pytest.mark.skipif(not (VJEPA2 / "src" / "masks").is_dir(), reason="src/vjepa2 submodule not initialised")


class _Tokens(nn.Module):
    """Frozen-encoder stand-in: tokens from pooled pixels of clip [t, t+1]; counts calls."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.proj = torch.randn(3, 1024, generator=torch.Generator().manual_seed(seed))
        self.calls = 0

    def forward_tokens(self, images):
        self.calls += 1
        batch, frames = images.shape[:2]
        pooled = F.avg_pool2d(images.flatten(0, 1), 16).flatten(2).transpose(1, 2) @ self.proj
        pooled = pooled.reshape(batch, frames, 512, 1024)
        return pooled[:, :-1] + pooled[:, 1:]


class _Flow(nn.Module):
    """FlowFormer++ stand-in returning 1/8-resolution flow [B, T-1, 2, 32, 64]."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        batch, frames = images.shape[:2]
        diff = (images[:, 1:] - images[:, :-1]).mean(dim=2).flatten(0, 1).unsqueeze(1)
        small = F.avg_pool2d(diff, 8).reshape(batch, frames - 1, 1, 32, 64)
        return torch.cat([small, -2 * small], dim=2)


class _Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1024, 1024)

    def forward(self, z_task_masked, U, context_indices, target_indices):
        base = self.proj(z_task_masked.mean(dim=1, keepdim=True))
        return base.expand(-1, target_indices.shape[1], -1) + 1e-3 * target_indices.unsqueeze(-1).float()


class _FakePipeline(nn.Module):
    forward = CaMoJEPAPipeline.forward
    _gather_tokens = staticmethod(CaMoJEPAPipeline._gather_tokens)

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(ablation_motion_branch=False, ablation_factorizer=False, ablation_confounder=False)
        self.context_encoder, self.target_encoder, self.flow_estimator = _Tokens(0), _Tokens(1), _Flow()
        self.flow_encoder = FlowTokenEncoder(2, 1024)
        self.fusion = GatedCrossAttentionFusion(1024, num_heads=16)
        self.factorizer = LatentFactorizer(1024, 512)
        self.confounder = ConfounderGRU(1024, 128)
        self.mask_sampler = MaskSampler(0.7, image_size=(512, 256), vjepa2_root=VJEPA2)
        self.predictor = _Predictor()


@needs_vjepa2
def test_score_window_runs_the_real_forward_once_per_mask_with_frozen_stages_computed_once(tmp_path):
    torch.manual_seed(0)
    pipeline = _FakePipeline().eval()
    original_sampler = pipeline.mask_sampler
    batch = FrameBatch(images=torch.rand(1, 16, 3, 256, 512, generator=torch.Generator().manual_seed(4)))
    means = accumulate_means(pipeline, [batch], "cpu")
    before = (pipeline.context_encoder.calls, pipeline.target_encoder.calls, pipeline.flow_estimator.calls)
    samplers = {"block": SeededBlockSampler(original_sampler), "random": RandomMaskSampler(4000, 1),
                "motion": MotionWeightedMaskSampler(4000, 2)}
    row, extras = score_window(pipeline, batch, samplers, means, flow_chunk=4, top_fraction=0.2, keep_tokens=True)

    after = (pipeline.context_encoder.calls, pipeline.target_encoder.calls, pipeline.flow_estimator.calls)
    assert after == (before[0] + 1, before[1] + 1, before[2] + 4)  # 15 pairs in chunks of 4
    assert pipeline.mask_sampler is original_sampler and isinstance(pipeline.context_encoder, _Tokens)
    assert extras["flow_shape"] == [1, 15, 2, 32, 64] and extras["motion_tokens_per_transition"] == 8
    assert row["masked_dynamic/random"][1] == row["masked_dynamic/motion"][1] == 4000
    assert row["masked_dynamic/motion"][0] > row["masked_dynamic/random"][0]
    for mask in MASKS:
        for target in TARGETS:
            for metric in METRICS:
                parts = {r: row[f"model/{mask}/{target}/{metric}/{r}"] for r in REGIONS}
                assert parts["dynamic"][1] + parts["static"][1] == parts["all"][1] == row[f"masked_dynamic/{mask}"][1]
                assert parts["all"][0] == pytest.approx(parts["dynamic"][0] + parts["static"][0])
    assert set(extras["tokens"]["masks"]) == set(MASKS) and len(extras["transition_flow_px"]) == 15

    series = WindowSeries()
    series.add_window(row)
    series.add_window(row)
    summary = summarize(series, n_boot=50, block=2, seed=0)
    random_dyn = summary["results"]["random"]["target_encoder"]["cos_ln"]["dynamic"]
    assert random_dyn["n_windows"] == 2 and random_dyn["ci95"] is None  # 2 windows < two blocks of 2
    assert random_dyn["ci95_iid"] == [pytest.approx(random_dyn["mean"])] * 2
    assert summary["paired_diff"]["random_minus_motion"]["target_encoder"]["cos"]["all"]["n_windows"] == 2
    assert len(headline_lines(summary)) == 1 + 3 * 3 + 3 + 3 + 3
    assert summary["paired_diff"]["gain_dynamic_minus_static"]["block"]["target_encoder"]["cos_ln"]["n_windows"] == 2
    _check_report_assembly(summary, extras, tmp_path)


def _check_report_assembly(summary: dict, extras: dict, root: Path) -> None:
    episodes = root / "episodes" / "test"
    episodes.mkdir(parents=True)
    frames = 40
    np.savez_compressed(episodes / "rec.npz", timestamps_us=np.arange(frames, dtype=np.int64) * 1_030_000,
                        image_paths=np.array([f"images/{i}.jpg" for i in range(frames)]),
                        can_bus=np.zeros((frames, 18), np.float32), ego_motion=np.zeros((frames, 2), np.float32))
    (root / "dataset_info.json").write_text(json.dumps({"camera": "CAM_P_F", "target_hz": 10.0}))
    config = CaMoJEPAConfig(dataset_root=str(root), dataset_split="test", stride=8)
    dataset = CaMoEpisodeDataset(root, split="test", history_length=16, stride=8)
    args = SimpleNamespace(data_root=str(root), split="test", train_hz=10.0, history_length=16, stride=8,
                           checkpoint_drive_id="drive-id", seed=0, top_fraction=0.2)
    header = {"task": "WM-12", "load_gates": {"camo": {"epoch": 2}, "flowformer": {"matched": 415, "expected": 415}}}
    identity = {name: {"path": name, "sha256": "0" * 64, "expected_sha256": None, "match": None}
                for name in ("checkpoint", "vitl", "flowformer")}
    report = build_report(header, summary, args, config, dataset, list(range(len(dataset))), extras,
                          masking={"M": 4000}, identity=identity, flow_diagnostics={"n": 0})
    text = json.dumps(report, ensure_ascii=False)
    assert report["data"]["n_windows"] == len(dataset) == 4 and report["data"]["camera"] == "CAM_P_F"
    assert "~1.03 s apart" in report["limitations"][0] and "stride 8" in report["limitations"][-1]
    assert "8 motion tokens per transition" in report["observations"]["notes"][0]
    assert report["region_rule"]["magnitude"].startswith("|flow| of the 32x64 flow") and "multiplied by 8" in text


def test_chunked_flow_equals_one_call():
    images = torch.rand(1, 16, 3, 256, 512, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(compute_flow(_Flow(), images, 4), compute_flow(_Flow(), images, 15))


def test_bootstrap_resamples_blocks_of_consecutive_windows_and_pools_by_tokens():
    idx = bootstrap_indices(10, 3, 200, seed=0)
    assert idx.shape == (200, 10) and idx.min() >= 0 and idx.max() < 10
    assert np.all((idx[:, 1] - idx[:, 0]) % 10 == 1) and np.all((idx[:, 2] - idx[:, 1]) % 10 == 1)
    assert bootstrap_indices(4, 5, 10, seed=0) is None  # fewer windows than two blocks
    pairs = np.array([[2.0, 4], [3.0, 6], [0.0, 0]])
    pooled = ratio_summary(pairs, {"ci95": bootstrap_indices(3, 1, 100, 1), "none": None})
    assert pooled["mean"] == 0.5 and pooled["n_tokens"] == 10 and pooled["n_windows"] == 2
    assert pooled["ci95"] == [0.5, 0.5] and pooled["none"] is None


def test_paired_difference_is_the_difference_of_the_pooled_means():
    # window means disagree in sign with the pooled means: random 0.5 x 800 + 0.2 x 900, motion 0.3 x 1500 + 0.5 x 100
    random_pairs = np.array([[0.5 * 800, 800], [0.2 * 900, 900]])
    motion_pairs = np.array([[0.3 * 1500, 1500], [0.5 * 100, 100]])
    resamples = {"ci95_iid": bootstrap_indices(2, 1, 200, 3)}
    diff = paired_summary(random_pairs, motion_pairs, resamples)
    pooled = ratio_summary(random_pairs, resamples)["mean"] - ratio_summary(motion_pairs, resamples)["mean"]
    assert diff["mean"] == pytest.approx(pooled) == pytest.approx(580 / 1700 - 500 / 1600)
    assert diff["mean"] > 0 and diff["windows_a_higher"] == 1 and diff["n_windows"] == 2


def test_contrast_is_the_signed_sum_of_pooled_means():
    a, b = np.array([[3.0, 4], [1.0, 2]]), np.array([[1.0, 4], [0.5, 2]])
    c, d = np.array([[2.0, 2], [2.0, 8]]), np.array([[0.0, 1], [4.0, 5]])
    resamples = {"ci95_iid": bootstrap_indices(2, 1, 100, 4)}
    gap = contrast_summary([(1.0, a), (-1.0, b), (-1.0, c), (1.0, d)], resamples)
    expected = (4 / 6 - 1.5 / 6) - (4 / 10 - 4 / 6)
    assert gap["mean"] == pytest.approx(expected) and gap["n_windows"] == 2
    assert paired_summary(a, b, resamples)["mean"] == pytest.approx(4 / 6 - 1.5 / 6)


def test_input_identity_and_gpu_arch_checks(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"weights")
    digest = sha256_file(path)
    assert verify_files({"w": (path, digest.upper())})["w"]["match"] is True
    assert verify_files({"w": (path, None)})["w"]["match"] is None
    with pytest.raises(IdentityError):
        verify_files({"w": (path, "0" * 64)})
    assert _arch_runs_on("sm_86", 8, 9) and _arch_runs_on("sm_60", 6, 0) and _arch_runs_on("compute_70", 8, 0)
    assert not _arch_runs_on("sm_90", 8, 9) and not _arch_runs_on("sm_75", 6, 0) and not _arch_runs_on("sm_61", 6, 0)


def test_heatmap_renders_under_500_kb(tmp_path):
    rng = np.random.default_rng(0)
    frame = tmp_path / "frame.jpg"
    Image.fromarray(rng.integers(0, 255, (1536, 1920, 3), dtype=np.uint8)).save(frame, quality=90)
    magnitude = torch.rand(15 * 512, generator=torch.Generator().manual_seed(6))
    mask = RandomMaskSampler(4000, seed=0)(torch.zeros(1, 15 * 512, 1))[2][0].numpy()
    size = render_mask_heatmap(frame, tmp_path / "heatmap.png", mask_indices=mask, cosines=rng.random(mask.size),
                               dynamic=dynamic_mask(magnitude).numpy(), magnitude=magnitude.numpy(), transition=7,
                               title="test", vmin=0.0, vmax=1.0)
    assert (tmp_path / "heatmap.png").stat().st_size == size < 500_000
