"""Command-line runner for the CaMo-JEPA pipeline training step."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

from .config import CaMoJEPAConfig
from .data import make_camo_dataloader
from .motion import MotionPretrainPipeline
from .perception import build_motion_encoder_adapter
from .pipeline import CaMoJEPAPipeline, save_checkpoint, train_step


def setup_vjepa2_path(vjepa2_root: str) -> None:
    """Add the V-JEPA2 root directory to sys.path for loading pretrained checkpoints."""
    path = Path(vjepa2_root).expanduser()

    if not path.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        path = (workspace_root / path).resolve()

    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    config = CaMoJEPAConfig()

    # Setup V-JEPA2 root path for loading pretrained checkpoints
    setup_vjepa2_path(config.vjepa2_root)

    # Create a dataloader for the CaMo dataset
    dataloader = make_camo_dataloader(config)
    batch = next(iter(dataloader))

    if config.stage == "motion_pretrain":
        # Create the motion pretraining pipeline model
        motion_pipeline = MotionPretrainPipeline(config.flow_channels, config.flow_token_dim)

        optimizer = torch.optim.AdamW(
            list(motion_pipeline.flow_encoder.parameters())
            + list(motion_pipeline.flow_decoder.parameters()),
            lr=config.learning_rate,
        )

        motion_pipeline.flow_estimator.eval()
        motion_pipeline.flow_encoder.train()
        motion_pipeline.flow_decoder.train()

        _, loss = motion_pipeline(batch.images)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save only the encoder weights; decoder is discarded after pre-training
        Path(config.motion_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(motion_pipeline.flow_encoder.state_dict(), config.motion_checkpoint_path)
        print({"status": "Motion pre-train step done", "loss": float(loss.detach())})
    elif config.stage == "camo_jepa":
        # CaMoJEPAPipeline loads pretrained motion weights internally via build_motion_encoder_adapter
        model = CaMoJEPAPipeline(config)

        # Filter trainable parameters (ignoring frozen ViT encoders)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

        # Execute full train step (Forward -> Loss -> Backprop -> Optimizer step -> EMA update)
        output = train_step(model, batch, optimizer, model.loss_fn)

        # Save checkpoint for trainable weights
        saved_path = save_checkpoint(
            checkpoint_path=config.predictor_output_checkpoint_path,
            model=model,
            epoch=1,
            optimizer=optimizer,
        )

        print({
            "status": "Train step completed successfully",
            "saved_checkpoint": str(saved_path),
            "fused_shape": tuple(output.fused_features.shape),
            "confounder_shape": tuple(output.U.shape),
            "jepa_pred_shape": tuple(output.z_pred.shape),
            "losses": {name: float(val.detach()) for name, val in output.losses.items()},
        })


if __name__ == "__main__":
    main()