#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import math
import random
import subprocess
import sys
from multiprocessing import Value
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.camo_jepa.config import CaMoJEPAConfig
from src.camo_jepa.contracts import FrameBatch
from src.camo_jepa.data.camo import make_camo_dataloader
from src.camo_jepa.pipeline.checkpoints import load_checkpoint
from src.camo_jepa.pipeline.phase1 import CaMoJEPAPipeline

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def move_batch_to_device(batch: FrameBatch, device: str) -> FrameBatch:
    batch.images = batch.images.to(device)
    if batch.ego_motion is not None:
        batch.ego_motion = batch.ego_motion.to(device)
    return batch


def reset_mask_sampler_counter(model: CaMoJEPAPipeline, value: int = -1) -> None:
    sampler = model.mask_sampler
    for _tp, gen in sampler._generator_cache.items():
        if hasattr(gen, "_itr_counter"):
            with gen._itr_counter.get_lock():
                gen._itr_counter.value = value


class ZeroHook:
    def __init__(self):
        self.call_count = 0

    def __call__(self, module, inputs, output):
        self.call_count += 1
        return torch.zeros_like(output)


class RandomFlowHook:
    def __init__(self, base_seed: int, mean: torch.Tensor, std: torch.Tensor):
        self.base_seed = base_seed
        self.mean = mean
        self.std = std
        self.call_count = 0

    def __call__(self, module, inputs, output):
        gen = torch.Generator(device=output.device)
        gen.manual_seed(self.base_seed + self.call_count)
        self.call_count += 1
        noise = torch.randn(output.shape, generator=gen,
                            dtype=output.dtype, device=output.device)
        return noise * self.std.to(output.device) + self.mean.to(output.device)


def layernorm_cosine_similarity(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    ln = nn.LayerNorm(z_pred.shape[-1], elementwise_affine=False,
                      device=z_pred.device, dtype=z_pred.dtype)
    z_pred_ln = ln(z_pred.float())
    z_target_ln = ln(z_target.float())
    return F.cosine_similarity(z_pred_ln, z_target_ln, dim=-1)


def classify_masked_tokens_by_flow(
    flow_maps: torch.Tensor,
    mask_indices: torch.Tensor,
    top_fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    B, T_1, C, H, W = flow_maps.shape
    pool_size = 2 
    
    mag = flow_maps.norm(dim=2)                          
    mag = mag.reshape(B * T_1, 1, H, W)
    mag_patches = F.avg_pool2d(mag, kernel_size=pool_size, stride=pool_size)
    
    grid_h, grid_w = H // pool_size, W // pool_size
    N_patches = grid_h * grid_w
    mag_patches = mag_patches.reshape(B, T_1 * N_patches)  

    all_mags = mag_patches.reshape(-1)
    k = max(1, int(math.ceil(top_fraction * all_mags.numel())))
    threshold = torch.topk(all_mags, k).values[-1].item()

    mag_at_mask = torch.gather(mag_patches, 1, mask_indices)  
    dynamic_mask = mag_at_mask > threshold
    static_mask = ~dynamic_mask

    return dynamic_mask, static_mask, threshold

def compute_delta_motion(z_fused: torch.Tensor, z_static: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
    B, T_1, N, D = z_fused.shape
    fused_flat = z_fused.reshape(B, T_1 * N, D)
    static_flat = z_static.reshape(B, T_1 * N, D)
    idx_exp = mask_indices.unsqueeze(-1).expand(-1, -1, D)
    fused_m = torch.gather(fused_flat, 1, idx_exp)    
    static_m = torch.gather(static_flat, 1, idx_exp)  
    diff_norm = (fused_m - static_m).norm(dim=-1)      
    static_norm = static_m.norm(dim=-1) + 1e-8          
    return diff_norm / static_norm


def circular_block_bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000, ci: float = 0.95, block_size: int | None = None, seed: int = 42) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(deltas)
    if n < 2:
        return {"mean": float(np.mean(deltas)), "ci_lower": float("nan"), "ci_upper": float("nan"), "se": float("nan"), "n_boot": n_boot, "block_size": 1}

    if block_size is None:
        block_size = max(1, int(np.ceil(n ** (1 / 3))))

    deltas_wrapped = np.concatenate([deltas, deltas[:block_size - 1]])
    boot_means = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block_size))

    for b in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        sample = np.concatenate([deltas_wrapped[s: s + block_size] for s in starts])[:n]
        boot_means[b] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {"mean": float(np.mean(deltas)), "ci_lower": lower, "ci_upper": upper, "se": float(np.std(boot_means, ddof=1)), "n_boot": n_boot, "block_size": block_size}

def autocast_ctx(args):
    if args.amp == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=dtype)


def generate_chart(summary: dict, bootstrap_results: dict, chart_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    variants = ["full", "static_only", "random_flow"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    width = 0.25
    x = np.arange(len(variants))

    glob_vals = [summary[v]["cos_ln_global"]["mean"] for v in variants]
    dyn_vals = [summary[v]["cos_ln_dynamic"]["mean"] for v in variants]
    stat_vals = [summary[v]["cos_ln_static"]["mean"] for v in variants]

    ax.bar(x - width, glob_vals, width, label='Global', color='#2196F3')
    ax.bar(x, dyn_vals, width, label='Dynamic (Top 20%)', color='#FF9800')
    ax.bar(x + width, stat_vals, width, label='Static (Bottom 80%)', color='#4CAF50')

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("LayerNorm Cosine Similarity")
    ax.set_title("Performance by Region")
    ax.legend()

    ax2 = axes[1]
    if "cos_ln_dynamic_full_vs_static" in bootstrap_results:
        g = bootstrap_results.get("cos_ln_global_full_vs_static", {})
        d = bootstrap_results.get("cos_ln_dynamic_full_vs_static", {})
        
        names = ["Global", "Dynamic"]
        means = [g.get("mean", 0), d.get("mean", 0)]
        yerr = [
            [g.get("mean", 0) - g.get("ci_lower", 0), g.get("ci_upper", 0) - g.get("mean", 0)],
            [d.get("mean", 0) - d.get("ci_lower", 0), d.get("ci_upper", 0) - d.get("mean", 0)]
        ]
        yerr = np.array(yerr).T

        ax2.bar(names, means, yerr=yerr, capsize=5, color=['#2196F3', '#FF9800'])
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title("Paired Delta (Full - Static_Only)\n(mean +/- 95% CI)")
        ax2.set_ylabel("Delta Cosine")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()


@torch.no_grad()
def phase1_baseline_and_partition(model: CaMoJEPAPipeline, dataloader, args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    device = args.device

    target_sum = torch.zeros(1024, device=device, dtype=torch.float64)
    target_count = 0
    flow_tok_sum, flow_tok_sq_sum, flow_tok_count = 0.0, 0.0, 0
    all_flow_maps = []
    
    calib_batches = args.calib_batches if not args.smoke_test else 2

    for i, batch in enumerate(tqdm(dataloader, desc="Phase 1: Baseline")):
        if i >= calib_batches:
            break

        batch = move_batch_to_device(batch, device)
        with autocast_ctx(args):
            out = model(batch)
            flow_maps = model.flow_estimator(batch.images)

        if out.target_patches is not None:
            tp = out.target_patches.float()
            target_sum += tp.sum(dim=(0, 1)).double()
            target_count += tp.shape[0] * tp.shape[1]

        if out.dynamic_features is not None:
            df = out.dynamic_features.float()
            flow_tok_sum += df.sum().item()
            flow_tok_sq_sum += (df ** 2).sum().item()
            flow_tok_count += df.numel()

        all_flow_maps.append(flow_maps.cpu())

    mean_target = (target_sum / target_count).float() if target_count > 0 else torch.zeros(1024, device=device)

    if flow_tok_count > 0:
        ft_mean = flow_tok_sum / flow_tok_count
        ft_var = max(0.0, flow_tok_sq_sum / flow_tok_count - ft_mean ** 2)
        ft_std = ft_var ** 0.5
    else:
        ft_mean, ft_std = 0.0, 1.0

    return {
        "mean_target": mean_target,
        "flow_tok_mean": torch.tensor(ft_mean),
        "flow_tok_std": torch.tensor(ft_std),
        "all_flow_maps": all_flow_maps,
        "top_fraction": args.top_fraction,
    }


@torch.no_grad()
def phase2_evaluate_variant(
    model: CaMoJEPAPipeline, dataloader, variant_name: str, phase1_data: dict[str, Any],
    args: argparse.Namespace, hook_fn: Any | None = None, hook_module: nn.Module | None = None,
) -> dict[str, Any]:
    device = args.device
    handle = hook_module.register_forward_hook(hook_fn) if (hook_fn and hook_module) else None

    model.eval()
    set_seed(args.seed)
    reset_mask_sampler_counter(model, value=-1)

    mean_target = phase1_data["mean_target"].to(device)
    all_flow_maps = phase1_data["all_flow_maps"]
    top_fraction = phase1_data["top_fraction"]

    window_cos_global, window_cos_dynamic, window_cos_static = [], [], []
    window_baseline_gap, window_delta_motion = [], []
    total_masked, total_dynamic, total_static = 0, 0, 0
    
    max_batches = args.max_batches if not args.smoke_test else 2

    for i, batch in enumerate(tqdm(dataloader, desc=f"Phase 2: {variant_name}")):
        if i >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        with autocast_ctx(args):
            out = model(batch)

        if out.z_pred is None or out.target_patches is None or out.mask_indices is None:
            continue

        cos_ln = layernorm_cosine_similarity(out.z_pred, out.target_patches)
        B, M = cos_ln.shape

        if i < len(all_flow_maps):
            flow_i = all_flow_maps[i].to(device)
            dyn_mask, sta_mask, _thr = classify_masked_tokens_by_flow(flow_i, out.mask_indices, top_fraction=top_fraction)
        else:
            dyn_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
            sta_mask = torch.ones(B, M, dtype=torch.bool, device=device)

        cos_flat = cos_ln.reshape(-1)
        window_cos_global.append(cos_flat.mean().item())

        if dyn_mask.any():
            window_cos_dynamic.append(cos_ln[dyn_mask].mean().item())
        if sta_mask.any():
            window_cos_static.append(cos_ln[sta_mask].mean().item())

        total_masked += int(B * M)
        total_dynamic += int(dyn_mask.sum().item())
        total_static += int(sta_mask.sum().item())

        if dyn_mask.any():
            mean_expanded = mean_target.unsqueeze(0).unsqueeze(0).expand_as(out.target_patches)
            cos_baseline = layernorm_cosine_similarity(mean_expanded, out.target_patches)
            gap = cos_ln[dyn_mask].mean().item() - cos_baseline[dyn_mask].mean().item()
            window_baseline_gap.append(gap)

        if out.fused_features is not None and out.static_features is not None:
            delta = compute_delta_motion(out.fused_features, out.static_features, out.mask_indices)
            if dyn_mask.any():
                window_delta_motion.append(delta[dyn_mask].mean().item())

    if handle is not None:
        handle.remove()

    return {
        "variant": variant_name,
        "cos_ln_global": window_cos_global,
        "cos_ln_dynamic": window_cos_dynamic,
        "cos_ln_static": window_cos_static,
        "baseline_gap_dynamic": window_baseline_gap,
        "delta_motion_dynamic": window_delta_motion,
        "total_masked_tokens": total_masked,
        "total_dynamic_tokens": total_dynamic,
        "total_static_tokens": total_static,
    }


def summarize_variant(raw: dict[str, Any]) -> dict[str, Any]:
    def _stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"mean": None, "std": None, "median": None, "n_windows": 0}
        a = np.array(vals)
        return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, "median": float(np.median(a)), "n_windows": len(a)}

    return {
        "cos_ln_global": _stats(raw["cos_ln_global"]),
        "cos_ln_dynamic": _stats(raw["cos_ln_dynamic"]),
        "cos_ln_static": _stats(raw["cos_ln_static"]),
        "baseline_gap_dynamic": _stats(raw["baseline_gap_dynamic"]),
        "delta_motion_dynamic": _stats(raw["delta_motion_dynamic"]),
        "total_masked_tokens": raw["total_masked_tokens"],
        "total_dynamic_tokens": raw["total_dynamic_tokens"],
        "total_static_tokens": raw["total_static_tokens"],
    }

def _serializable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except:
        return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Two-pass ablation: Static vs Dynamic flow for CaMo-JEPA")
    
    # arg chuẩn
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="dataset_camo/vinfast")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--chart-output", default=None, help="Chart output path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    
    # arg điều khiển mô hình 
    parser.add_argument("--vitl-checkpoint", default=None)
    parser.add_argument("--flowformer-checkpoint", default=None)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--calib-batches", type=int, default=20)

    # Các arg thuật toán
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    if "cuda" in args.device and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("=" * 60)
    logger.info("CaMo-JEPA Two-Pass Ablation Evaluation")
    logger.info("=" * 60)

    config_kwargs: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "dataset_split": args.dataset_split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": not args.no_shuffle,
        "ablation_motion_branch": False,  
    }
    if args.vitl_checkpoint: config_kwargs["vitl_checkpoint_path"] = args.vitl_checkpoint
    if args.flowformer_checkpoint: config_kwargs["motion_estimator_checkpoint_path"] = args.flowformer_checkpoint

    config = CaMoJEPAConfig(**config_kwargs)
    model = CaMoJEPAPipeline(config)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.to(args.device)
    model.eval()

    dataloader = make_camo_dataloader(config)

    # Phase 1
    logger.info("Phase 1: Computing baseline & flow partitioning ...")
    phase1_data = phase1_baseline_and_partition(model, dataloader, args)
    logger.info(f"  Flow-token distribution: mean={phase1_data['flow_tok_mean'].item():.4f}, std={phase1_data['flow_tok_std'].item():.4f}")

    # Phase 2
    logger.info("Phase 2: Evaluating variants with fixed mask seed ...")
    raw_full = phase2_evaluate_variant(model, dataloader, "full", phase1_data, args)
    
    zero_hook = ZeroHook()
    raw_static = phase2_evaluate_variant(model, dataloader, "static_only", phase1_data, args, hook_fn=zero_hook, hook_module=model.flow_encoder)

    random_hook = RandomFlowHook(base_seed=args.seed + 1000, mean=phase1_data["flow_tok_mean"], std=phase1_data["flow_tok_std"])
    raw_random = phase2_evaluate_variant(model, dataloader, "random_flow", phase1_data, args, hook_fn=random_hook, hook_module=model.flow_encoder)

    # Summarize
    summary = {
        "full": summarize_variant(raw_full),
        "static_only": summarize_variant(raw_static),
        "random_flow": summarize_variant(raw_random),
    }

    bootstrap_results = {}
    n_windows = min(len(raw_full["cos_ln_global"]), len(raw_static["cos_ln_global"]))
    if n_windows >= 2:
        paired_cos_global = np.array(raw_full["cos_ln_global"][:n_windows]) - np.array(raw_static["cos_ln_global"][:n_windows])
        bootstrap_results["cos_ln_global_full_vs_static"] = circular_block_bootstrap_ci(paired_cos_global, n_boot=args.n_boot, block_size=args.block_size, seed=args.seed)

    n_dyn_windows = min(len(raw_full["cos_ln_dynamic"]), len(raw_static["cos_ln_dynamic"]))
    if n_dyn_windows >= 2:
        paired_cos_dyn = np.array(raw_full["cos_ln_dynamic"][:n_dyn_windows]) - np.array(raw_static["cos_ln_dynamic"][:n_dyn_windows])
        bootstrap_results["cos_ln_dynamic_full_vs_static"] = circular_block_bootstrap_ci(paired_cos_dyn, n_boot=args.n_boot, block_size=args.block_size, seed=args.seed)

    # Output JSON & Chart
    results = {
        "experiment": "ablation_twopass_static_dynamic_flow",
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "checkpoint": args.checkpoint,
        "dataset": args.dataset_root,
        "device": args.device,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "config": {k: v for k, v in vars(args).items() if _serializable(v)},
        "phase1": {
            "flow_token_distribution": {"mean": phase1_data["flow_tok_mean"].item(), "std": phase1_data["flow_tok_std"].item()},
            "top_fraction": args.top_fraction,
        },
        "methodology_notes": {
            "mask_seed_control": "MaskSampler._itr_counter is reset to -1 before each variant.",
            "metrics_scope": "All cosine similarity metrics are computed ONLY on masked tokens.",
            "cos_ln": "LayerNorm Cosine Similarity: z_pred and z_target are LayerNorm-ed before cosine computation.",
            "delta_motion": "delta_motion = ||z_fused[mask] - z_static[mask]||_2 / (||z_static[mask]||_2 + eps).",
            "dynamic_classification": f"Dynamic tokens = top {args.top_fraction*100:.0f}% by optical flow magnitude.",
        },
        "variants": summary,
        "bootstrap_95ci": bootstrap_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=str)
    
    chart_output = args.chart_output or str(out_path.with_suffix("")) + "_chart.png"
    generate_chart(summary, bootstrap_results, chart_output)

    logger.info(f"Results saved to {out_path}")
    logger.info(f"Chart saved to {chart_output}")

    print("\n" + "=" * 70)
    print("  TWO-PASS ABLATION RESULTS")
    print("=" * 70)
    for vname in ["full", "static_only", "random_flow"]:
        v = summary[vname]
        g = v["cos_ln_global"]["mean"]
        d = v["cos_ln_dynamic"]["mean"]
        s = v["cos_ln_static"]["mean"]
        print(f"  {vname:15s}  cos_ln: Global={g if g else 'N/A':>7.4f}  Dynamic={d if d else 'N/A':>7.4f}  Static={s if s else 'N/A':>7.4f}")

    if "cos_ln_dynamic_full_vs_static" in bootstrap_results:
        ci = bootstrap_results["cos_ln_dynamic_full_vs_static"]
        print(f"\n  Paired Δ(full-static) on Dynamic tokens:")
        print(f"    mean={ci['mean']:.4f}  95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print("=" * 70)

if __name__ == "__main__":
    main()