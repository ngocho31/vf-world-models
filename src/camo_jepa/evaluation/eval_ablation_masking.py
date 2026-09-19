"""WM-12 masking ablation: how well the predictor recovers masked latent tokens, dynamic vs static.

Three mask strategies run on the same windows of a trained CaMo-JEPA
checkpoint: ``block`` (the train-time multiblock sampler, as reference),
``random``, and ``motion`` (probability proportional to optical flow); the last
two mask the same number of tokens ``M`` that ``block`` masks on average.
Scores are cosines between ``ModelOutput.z_pred`` and ``z*`` on masked tokens
only, split into the dynamic region (top 20% patches by FlowFormer++ ``|flow|``)
and the rest.

Run from the repository root (the encoders import ``src``)::

    python3 -m src.camo_jepa.evaluation.eval_ablation_masking \\
        --data-root dataset_camo/vf_tar --split test --checkpoint best_camo.pt --out outputs/wm12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..cli import setup_vjepa2_path
from ..config import CaMoJEPAConfig
from ..data import CaMoEpisodeDataset, collate_camo_samples
from . import eval_model_loading as loading
from .eval_environment import IdentityError, git_commit, resolve_device, verify_files
from .mask_heatmap import render_heatmaps
from .masking_report import build_report, headline_lines, summarize
from .masking_strategies import MotionWeightedMaskSampler, RandomMaskSampler, SeededBlockSampler, measure_block_mask_fraction
from .masking_window_runner import accumulate_means, score_window, to_device, window_frame
from .window_bootstrap import WindowSeries

REPORT_NAME = "wm12_ablation_masking.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", required=True, help="converted CaMo dataset root (episodes/<split>/*.npz)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True, help="CaMo-JEPA checkpoint, e.g. best_camo.pt")
    parser.add_argument("--checkpoint-drive-id", default=None, help="provenance only")
    parser.add_argument("--expect-epoch", type=int, default=None, help="fail unless the checkpoint stores this epoch")
    parser.add_argument("--vitl", default=CaMoJEPAConfig.vitl_checkpoint_path)
    parser.add_argument("--flowformer", default=CaMoJEPAConfig.motion_estimator_checkpoint_path)
    for name in ("checkpoint", "vitl", "flowformer"):
        parser.add_argument(f"--{name}-sha256", default=None, help=f"fail before loading unless --{name} has this sha256")
    for flag in ("motion", "confounder", "factorizer"):
        parser.add_argument(f"--ablation-{flag}", action="store_true", help=f"checkpoint trained without the {flag} branch")
    parser.add_argument("--history-length", type=int, default=CaMoJEPAConfig.history_length)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=None, help="evenly spaced subset, for smoke runs")
    parser.add_argument("--top-fraction", type=float, default=0.2, help="dynamic share of patches per transition")
    parser.add_argument("--block-draws", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-block", type=int, default=5, help="consecutive windows per bootstrap block")
    parser.add_argument("--heatmap-windows", type=int, default=6)
    parser.add_argument("--flow-chunk", type=int, default=15, help="frame pairs per FlowFormer++ call")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-hz", type=float, default=10.0, help="frame rate of the training data (provenance)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> CaMoJEPAConfig:
    return CaMoJEPAConfig(
        dataset_root=args.data_root, dataset_split=args.split, history_length=args.history_length, stride=args.stride,
        batch_size=1, shuffle=False, num_workers=args.num_workers, vitl_checkpoint_path=args.vitl,
        motion_estimator_checkpoint_path=args.flowformer, ablation_motion_branch=args.ablation_motion,
        ablation_confounder=args.ablation_confounder, ablation_factorizer=args.ablation_factorizer)


def load_and_gate(args, config, device, gates: dict):
    """Build the pipeline and run the three load gates; raises ``GateError`` on any gap."""
    pipeline = loading.build_pipeline(config, device)
    gates["camo"] = loading.load_camo_checkpoint(pipeline, args.checkpoint, args.expect_epoch)
    gates["flowformer"] = loading.check_flowformer_weights(pipeline.flow_estimator.backbone, args.flowformer)
    gates["vit"] = loading.check_vit_weights(pipeline, args.vitl)
    print("[gate] CaMo modules, FlowFormer++ and both ViT-L encoders hold their checkpoint tensors", flush=True)
    return pipeline


def make_loader(config: CaMoJEPAConfig, args) -> tuple[CaMoEpisodeDataset, list[int], DataLoader]:
    dataset = CaMoEpisodeDataset(config.dataset_root, split=config.dataset_split, history_length=config.history_length,
                                 stride=config.stride, image_size=config.image_size)
    count = len(dataset) if args.max_windows is None else min(args.max_windows, len(dataset))
    window_ids = sorted({int(i) for i in np.linspace(0, len(dataset) - 1, count).round()})
    loader = DataLoader(Subset(dataset, window_ids), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_camo_samples)
    return dataset, window_ids, loader


def score_windows(pipeline, dataset, window_ids, loader, samplers, means, args, device):
    """Pass 2 over every window; returns the series, per-window rows, heatmap data and last extras."""
    count = min(args.heatmap_windows, len(window_ids))
    shown = {window_ids[int(i)] for i in np.linspace(0, len(window_ids) - 1, count).round()} if count else set()
    series, per_window, captured, extras = WindowSeries(), [], {}, {}
    for position, batch in enumerate(loader):
        window_id, tick = window_ids[position], time.time()
        row, extras = score_window(pipeline, to_device(batch, device), samplers, means, args.flow_chunk,
                                   args.top_fraction, keep_tokens=window_id in shown)
        series.add_window(row)
        if "tokens" in extras:
            captured[window_id] = extras.pop("tokens")
        start = dataset._windows[window_id].start_index
        per_window.append({"window": window_id, "start_frame": start, "first_frame": window_frame(dataset, window_id, 0).stem,
                           "transition_flow_px": extras["transition_flow_px"], "row": row})
        print(f"[window {position + 1}/{len(window_ids)}] id {window_id} start {start}: {time.time() - tick:.1f}s", flush=True)
    return series, per_window, captured, extras


def flow_diagnostics(per_window: list[dict]) -> dict:
    """How often a transition barely moves, which makes its 'dynamic' patches noise."""
    flows = np.array([f for window in per_window for f in window["transition_flow_px"]])
    return {"unit": "mean |flow| of a transition in input pixels, over window-transitions (overlaps counted)",
            "n": int(flows.size), "median_px": float(np.median(flows)),
            "below_2px": int((flows < 2).sum()), "below_5px": int((flows < 5).sum())}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started, out_dir = time.time(), Path(args.out)
    (out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = build_config(args)
    setup_vjepa2_path(config.vjepa2_root)
    gates: dict = {}
    header = {"task": "WM-12 [Nhóm 1][Ablation] Masking", "git_commit": git_commit(),
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "environment": {"torch": torch.__version__, "python": sys.version.split()[0],
                              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
              "load_gates": gates}
    try:
        identity = verify_files({"checkpoint": (args.checkpoint, args.checkpoint_sha256),
                                 "vitl": (args.vitl, args.vitl_sha256), "flowformer": (args.flowformer, args.flowformer_sha256)})
        pipeline = load_and_gate(args, config, device, gates)
        dataset, window_ids, loader = make_loader(config, args)

        def token_gate(context_tokens, target_tokens):
            gates["token_l2"] = loading.check_token_norms(context_tokens, target_tokens)

        means = accumulate_means(pipeline, loader, device, on_first_window=token_gate)
    except (IdentityError, loading.GateError) as error:
        header["failed_gate"] = {"gate": getattr(error, "gate", "identity"), "report": error.report}
        (out_dir / REPORT_NAME).write_text(json.dumps(header, indent=2, default=str, ensure_ascii=False))
        raise
    print(f"[gate] token L2 {gates['token_l2']}; pass 1 done at {time.time() - started:.0f}s", flush=True)

    tokens_per_window = (config.history_length - 1) * (config.image_size[1] // 16) * (config.image_size[0] // 16)
    block_fraction = measure_block_mask_fraction(pipeline.mask_sampler, config.history_length - 1, args.block_draws, args.seed)
    num_masked = round(block_fraction["mean"] * tokens_per_window)
    print(f"[mask] block masks {block_fraction['mean']:.4f} of {tokens_per_window} tokens -> M = {num_masked}", flush=True)
    samplers = {"block": SeededBlockSampler(pipeline.mask_sampler, seed=args.seed),
                "random": RandomMaskSampler(num_masked, seed=args.seed + 1),
                "motion": MotionWeightedMaskSampler(num_masked, seed=args.seed + 2)}
    series, per_window, captured, extras = score_windows(pipeline, dataset, window_ids, loader, samplers, means, args, device)

    summary = summarize(series, args.n_boot, args.bootstrap_block, args.seed)
    report = build_report(header, summary, args, config, dataset, window_ids, extras,
                          masking={"mask_ratio": config.mask_ratio, "block_mask_fraction": block_fraction,
                                   "M": num_masked, "tokens_per_window": tokens_per_window},
                          identity=identity, flow_diagnostics=flow_diagnostics(per_window))
    (out_dir / "wm12_per_window.json").write_text(json.dumps(per_window))
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n".join(headline_lines(summary)), flush=True)
    transition = (config.history_length - 1) // 2
    report["heatmaps"] = render_heatmaps(captured, lambda w: window_frame(dataset, w, transition), out_dir / "heatmaps", transition)
    report["runtime_s"] = round(time.time() - started, 1)
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
