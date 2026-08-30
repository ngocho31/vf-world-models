"""Command-line runner for CaMo-JEPA training."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch

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
    config = CaMoJEPAConfig()

    # Setup V-JEPA2 root path
    setup_vjepa2_path(config.vjepa2_root)

    # Determine the active Ablation Study scenario
    scenario_name, ablation_flags = get_scenario_info(config)
    log_dir = Path(config.output_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"metrics_{scenario_name}.jsonl"

    checkpoint_dir = Path(config.output_checkpoint_path).expanduser().parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = Path(config.output_checkpoint_path).expanduser()
    best_ckpt_path = checkpoint_dir / f"best_{latest_ckpt_path.name}"

    print("=" * 70)
    print(f"[INFO] Running Scenario : {scenario_name}")
    print(f"[INFO] Ablation Flags : {ablation_flags}")
    print(f"[INFO] Logging metrics to: {log_file_path}")
    print(f"[INFO] Latest Checkpoint : {latest_ckpt_path}")
    print(f"[INFO] Best Checkpoint   : {best_ckpt_path}")
    print("=" * 70)

    # Dataloader & Device Setup
    dataloader = make_camo_dataloader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU Device: {torch.cuda.get_device_name(0)}")

    # Pipeline Model
    model = CaMoJEPAPipeline(config)
    model = model.to(device)

    # Optimizer Parameter Grouping
    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
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
            model=model,
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
            output = train_step(model, batch, optimizer, model.loss_fn)
            step_time = time.time() - step_start_time

            current_losses = {
                k: float(v.detach().cpu()) for k, v in output.losses.items()
            }
            step_loss = current_losses.get("total", 0.0)
            running_epoch_loss += step_loss
            num_batches_in_epoch += 1

            if batch_idx % n_steps_per_epoch == 0 or batch_idx == total_steps:
                loss_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in current_losses.items()]
                )
                print(
                    f"[Epoch {epoch:03d}/{num_epochs:03d}][Step {batch_idx:04d}/{total_steps:04d}] "
                    f"Loss: {step_loss:.4f} ({loss_str}) | "
                    f"Speed: {step_time:.2f}s/step"
                )
                step_log_record = {
                    "timestamp": datetime.now().isoformat(),
                    "epoch": epoch,
                    "step": batch_idx,
                    "total_steps": total_steps,
                    "scenario": scenario_name,
                    "losses": current_losses,
                    "step_time_sec": round(step_time, 3),
                }
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(step_log_record) + "\n")

                # Save checkpoint for the current step
                step_ckpt_path = checkpoint_dir / f"step_{batch_idx:06d}.pt"
                saved_path = save_checkpoint(
                    checkpoint_path=step_ckpt_path,
                    model=model,
                    epoch=epoch,
                    optimizer=optimizer,
                )
                print(f"[INFO] Mid-epoch checkpoint saved to {step_ckpt_path}")

        # Compute the average loss for the epoch
        epoch_avg_loss = running_epoch_loss / max(1, num_batches_in_epoch)

        # Save checkpoint for the latest state of the model
        saved_path = save_checkpoint(
            checkpoint_path=latest_ckpt_path,
            model=model,
            epoch=epoch,
            optimizer=optimizer,
        )
        print(f"[INFO] [Epoch {epoch:03d}] Latest checkpoint updated at {saved_path}")

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            saved_path = save_checkpoint(
                checkpoint_path=best_ckpt_path,
                model=model,
                epoch=epoch,
                optimizer=optimizer,
            )
            print(f"[INFO] [Epoch {epoch:03d}] New best loss ({best_loss:.4f})! Saved best checkpoint to {best_ckpt_path}")

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


if __name__ == "__main__":
    main()