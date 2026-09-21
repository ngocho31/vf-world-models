"""Command-line runner for CaMo-JEPA training.

OPTIMIZATION NOTES:
- [OPT-3] Upgraded from torch.nn.DataParallel to DistributedDataParallel (DDP)
  via `torchrun --standalone --nproc_per_node=gpu`. DDP runs each GPU as a
  separate process, keeping tensors contiguous in local GPU memory — this
  eliminates the CUDNN_STATUS_NOT_SUPPORTED error from FlowFormer grid_sample.
  Falls back gracefully to single-GPU when torchrun is not used.
- [OPT-1] Initialized torch.cuda.amp.GradScaler and passed it to train_step()
  to work in conjunction with the bfloat16 autocast added in engine.py.
- [OPT-4] Added torch.backends.cudnn.benchmark = True to let cuDNN auto-tune
  kernel selection for fixed input sizes (free ~5-10% speedup).
- [OPT-2] Only rank-0 process writes checkpoints and log files to avoid
  concurrent writes and duplicate metrics entries.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

# Set before importing torch: reduces GPU memory fragmentation (recommended by PyTorch OOM error messages)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from .config import CaMoJEPAConfig
from .data import make_camo_dataloader
from .pipeline import CaMoJEPAPipeline, save_checkpoint, load_checkpoint, train_step


def setup_vjepa2_path(vjepa2_root: str) -> None:
    """Add the V-JEPA2 root directory to sys.path for loading pretrained checkpoints."""
    path = Path(vjepa2_root).expanduser()

    if not path.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        path = (workspace_root / path).resolve()

    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def get_scenario_info(config: CaMoJEPAConfig) -> tuple[str, dict[str, bool]]:
    """Determine the active Ablation Study scenario name and config breakdown."""
    ablation_flags = {
        "motion_branch": config.ablation_motion_branch,
        "confounder": config.ablation_confounder,
        "factorizer": config.ablation_factorizer,
    }
    scenario_name = "Full_CaMo_JEPA" if not any(ablation_flags.values()) else f"Ablation_{'_'.join([k for k, v in ablation_flags.items() if v])}"
    return scenario_name, ablation_flags


def main() -> None:
    # ── DDP / Single-GPU detection ─────────────────────────────────────────────
    # LOCAL_RANK is injected by `torchrun`. When running plain `python3`, it is
    # absent so we default to -1, which means "no distributed init, single GPU".
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0
    is_main = (not is_ddp) or (local_rank == 0)  # Only rank-0 writes files/logs

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = CaMoJEPAConfig()

    # Setup V-JEPA2 root path
    setup_vjepa2_path(config.vjepa2_root)

    # [OPT-4] Enable cuDNN auto-tuner for fixed input sizes
    torch.backends.cudnn.benchmark = True

    # Determine the active Ablation Study scenario
    scenario_name, ablation_flags = get_scenario_info(config)
    log_dir = Path(config.output_log_dir)
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"metrics_{scenario_name}.jsonl"

    checkpoint_dir = Path(config.output_checkpoint_path).expanduser().parent
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = Path(config.output_checkpoint_path).expanduser()
    best_ckpt_path = checkpoint_dir / f"best_{latest_ckpt_path.name}"

    if is_main:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        mode = f"DDP across {dist.get_world_size()} GPUs" if is_ddp else "Single-GPU"
        print("=" * 70)
        print(f"[INFO] Running Scenario : {scenario_name}")
        print(f"[INFO] Ablation Flags   : {ablation_flags}")
        print(f"[INFO] Training Mode    : {mode}")
        print(f"[INFO] Logging metrics to: {log_file_path}")
        print(f"[INFO] Latest Checkpoint : {latest_ckpt_path}")
        print(f"[INFO] Best Checkpoint   : {best_ckpt_path}")
        print("=" * 70)

    if device.type == "cuda":
        gpu_idx = local_rank if is_ddp else 0
        props = torch.cuda.get_device_properties(gpu_idx)
        total_vram_gb = props.total_memory / (1024 ** 3)
        if is_main:
            print(f"[INFO] GPU Device       : {torch.cuda.get_device_name(gpu_idx)}")
            print(f"[INFO] Number of GPUs   : {torch.cuda.device_count()} (active in job)")
            print(f"[INFO] Total VRAM (HW)  : {total_vram_gb:.2f} GB (total physical VRAM per GPU)")
            print(f"[INFO] VRAM Free        : {torch.cuda.mem_get_info(gpu_idx)[0] / (1024**3):.2f} GB (before model load)")

    # ── Dataloader with DistributedSampler when DDP ────────────────────────────
    from .data.camo import CaMoEpisodeDataset, collate_camo_samples
    from torch.utils.data import DataLoader
    dataset = CaMoEpisodeDataset(
        dataset_root=config.dataset_root,
        split=config.dataset_split,
        history_length=config.history_length,
        stride=config.stride,
        image_size=config.image_size,
        max_cached_episodes=config.max_cached_episodes,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),  # shuffle=False when using DistributedSampler
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_camo_samples,
    )

    # ── Model: build → move to GPU → optionally wrap DDP ──────────────────────
    model = CaMoJEPAPipeline(config)
    model = model.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb  = torch.cuda.memory_reserved(device) / (1024 ** 3)
        free_gb      = torch.cuda.mem_get_info(device.index or 0)[0] / (1024 ** 3)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if is_main:
            print(f"[INFO] VRAM after model load:")
            print(f"  - Allocated (model weights) : {allocated_gb:.2f} GB")
            print(f"  - Reserved (PyTorch cache)  : {reserved_gb:.2f} GB")
            print(f"  - Free (still available)    : {free_gb:.2f} GB")
            print(f"  - Total (HW)                : {total_vram_gb:.2f} GB")

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,  # Required for Ablation studies where some modules are bypassed in forward()
        )

    # [OPT-1] Initialize GradScaler for AMP (works with bfloat16 autocast in engine.py)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # base_model: unwrap DDP/DP to access model's own methods (EMA, loss_fn, etc.)
    base_model = getattr(model, "module", model)

    # Optimizer Parameter Grouping
    decay_params = []
    no_decay_params = []

    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply no weight decay to bias and LayerNorm/BatchNorm
        if "bias" in name or "norm" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": getattr(config, "weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # Resume from Checkpoint
    start_epoch = 1
    best_loss = float("inf")

    if latest_ckpt_path.exists() and not config.pretrained:
        print(f"[INFO] Found checkpoint at {latest_ckpt_path}. Loading weights...")
        ckpt_data, epoch, optimizer = load_checkpoint(
            checkpoint_path=latest_ckpt_path,
            model=base_model,  # [OPT-2] Always load into the base model
            optimizer=optimizer,
            strict=False,
        )
        if epoch is not None:
            print(f"[INFO] Resuming training from epoch {epoch + 1}.")
            start_epoch = epoch + 1
            if "best_loss" in ckpt_data:
                best_loss = float(ckpt_data["best_loss"])
    else:
        print(f"[INFO] No checkpoint found at {latest_ckpt_path} or pretrained mode enabled. Starting from scratch.")

    num_epochs = config.num_epochs
    n_steps_per_epoch = config.n_steps_per_epoch
    total_steps = len(dataloader)

    for epoch in range(start_epoch, num_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)  # Ensures different shuffle per epoch in DDP
        model.train()
        running_epoch_loss = 0.0
        num_batches_in_epoch = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            step_start_time = time.time()
            # Execute full train step (Forward -> Loss -> Backprop -> Optimizer step -> EMA update)
            if hasattr(batch, "to"):
                batch = batch.to(device)
            elif hasattr(batch, "images"):
                batch.images = batch.images.to(device)
            output = train_step(model, batch, optimizer, base_model.loss_fn, scaler=scaler)
            step_time = time.time() - step_start_time

            current_losses = {
                k: float(v.detach().cpu()) for k, v in output.losses.items()
            }
            step_loss = current_losses.get("total", 0.0)
            running_epoch_loss += step_loss
            num_batches_in_epoch += 1

            if batch_idx % n_steps_per_epoch == 0 or batch_idx == total_steps:
                # Only rank-0 prints and writes — avoids duplicate lines & concurrent file writes
                if is_main:
                    loss_str = " | ".join(
                        [f"{k}: {v:.4f}" for k, v in current_losses.items()]
                    )
                    
                    # Gradient Diagnostics (Requires scaler.unscale_ to be called in engine.py first)
                    gate_grad_norm = 0.0
                    proj_grad_norm = 0.0
                    predictor_grad_norm = 0.0

                    if not config.ablation_motion_branch and base_model.fusion.gate.grad is not None:
                        gate_grad_norm = base_model.fusion.gate.grad.norm().item()
                    if not config.ablation_factorizer and hasattr(base_model.factorizer, 'raw_projection') and base_model.factorizer.raw_projection.grad is not None:
                        proj_grad_norm = base_model.factorizer.raw_projection.grad.norm().item()
                    if hasattr(base_model, 'predictor'):
                        pred_grads = [p.grad for p in base_model.predictor.parameters() if p.grad is not None]
                        if pred_grads:
                            predictor_grad_norm = torch.linalg.norm(
                                torch.stack([torch.linalg.norm(g.detach()) for g in pred_grads])
                            ).item()
                    
                    print(
                        f"[Epoch {epoch:03d}/{num_epochs:03d}][Step {batch_idx:04d}/{total_steps:04d}] "
                        f"Loss: {step_loss:.4f} ({loss_str}) | "
                        f"Speed: {step_time:.2f}s/step\n"
                        f"      -> Grad Norms | Predictor: {predictor_grad_norm:.4e} | "
                        f"Fusion Gate: {gate_grad_norm:.4e} | Factorizer Proj: {proj_grad_norm:.4e}"
                    )
                    step_log_record = {
                        "timestamp": datetime.now().isoformat(),
                        "epoch": epoch,
                        "step": batch_idx,
                        "total_steps": total_steps,
                        "scenario": scenario_name,
                        "losses": current_losses,
                        "step_time_sec": round(step_time, 3),
                        "grad_norms": {
                            "predictor": predictor_grad_norm,
                            "fusion_gate": gate_grad_norm,
                            "factorizer_proj": proj_grad_norm,
                        }
                    }
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(step_log_record) + "\n")

                    # Save mid-epoch checkpoint (include epoch to prevent overwriting)
                    step_ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}_step_{batch_idx:06d}.pt"
                    save_checkpoint(
                        checkpoint_path=step_ckpt_path,
                        model=base_model,  # Always save base model state_dict (unwrapped from DDP)
                        epoch=epoch,
                        optimizer=optimizer,
                    )
                    print(f"[INFO] Mid-epoch checkpoint saved to {step_ckpt_path}")

        # Compute the average loss for the epoch
        epoch_avg_loss = running_epoch_loss / max(1, num_batches_in_epoch)

        # ── Epoch-end parameter tracking (rank-0 only) ────────────────────────
        if is_main:
            # 1. Fusion gate: tanh(gate) controls how much motion info is blended in
            if not config.ablation_motion_branch:
                gate_val = torch.tanh(base_model.fusion.gate).item()
            else:
                gate_val = 0.0

            # 2. LatentFactorizer: sanity-check the orthogonal projector P_task
            if not config.ablation_factorizer:
                with torch.no_grad():
                    p_task_fp32 = base_model.factorizer._compute_p_task(dtype=torch.float32)
                    trace_val       = torch.trace(p_task_fp32).item()
                    rank_val        = torch.linalg.matrix_rank(p_task_fp32).item()
                    idempotent_err  = torch.norm(p_task_fp32 @ p_task_fp32 - p_task_fp32).item()
            else:
                trace_val, rank_val, idempotent_err = 0.0, 0, 0.0

            print(
                f"[Epoch {epoch:03d} TRACKING] "
                f"Fusion Gate (tanh): {gate_val:.4f} | "
                f"P_task -> Trace: {trace_val:.4f}/{base_model.factorizer.task_dim if not config.ablation_factorizer else 'N/A'} | "
                f"Rank: {rank_val} | "
                f"P²-P Error: {idempotent_err:.2e}"
            )

        # Only rank-0 saves checkpoints to avoid race conditions
        if is_main:
            saved_path = save_checkpoint(
                checkpoint_path=latest_ckpt_path,
                model=base_model,  # Always save unwrapped base model state_dict
                epoch=epoch,
                optimizer=optimizer,
            )
            print(f"[INFO] [Epoch {epoch:03d}] Latest checkpoint updated at {saved_path}")

            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                saved_path = save_checkpoint(
                    checkpoint_path=best_ckpt_path,
                    model=base_model,
                    epoch=epoch,
                    optimizer=optimizer,
                )
                print(f"[INFO] [Epoch {epoch:03d}] New best loss ({best_loss:.4f})! Saved best checkpoint to {best_ckpt_path}")

    if is_main:
        print({
            "status": "Train step completed successfully",
            "latest_checkpoint": str(latest_ckpt_path),
            "best_checkpoint": str(best_ckpt_path),
            "best_loss": best_loss,
            "static_features_shape": tuple(output.static_features.shape),
            "target_patches_shape": tuple(output.target_patches.shape),
            "dynamic_features_shape": tuple(output.dynamic_features.shape) if output.dynamic_features is not None else None,
            "fused_features_shape": tuple(output.fused_features.shape) if output.fused_features is not None else None,
            "confounder_features_shape": tuple(output.U.shape) if output.U is not None else None,
            "z_task_shape": tuple(output.z_task.shape) if output.z_task is not None else None,
            "z_exogenous_shape": tuple(output.z_exogenous.shape) if output.z_exogenous is not None else None,
            "P_task_shape": tuple(output.P_task.shape) if output.P_task is not None else None,
            "P_exogenous_shape": tuple(output.P_exogenous.shape) if output.P_exogenous is not None else None,
            "jepa_predicted_features_shape": tuple(output.z_pred.shape),
            "final_losses": {name: float(val.detach()) for name, val in output.losses.items()},
        })

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()