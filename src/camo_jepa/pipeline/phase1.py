"""CaMo-JEPA pipeline: Static + Motion + Fusion + Factorization + Confounder + Predictor."""

from __future__ import annotations

import torch
from torch import nn

from ..causal import ConfounderGRU, LatentFactorizer
from ..config import CaMoJEPAConfig
from ..contracts import FrameBatch, ModelOutput
from ..evaluation import CaMoJEPALoss
from ..perception import (
    CausalPredictor,
    MaskSampler,
    ResidualFusion,
    build_vit_encoders,
    build_motion_encoder_adapter,
)


class CaMoJEPAPipeline(nn.Module):
    """Runs each stage explicitly while keeping stages replaceable."""

    def __init__(self, config: CaMoJEPAConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Static Branch (Load ViT-L weights)
        self.context_encoder, self.target_encoder = build_vit_encoders(
            checkpoint_path=config.vitl_checkpoint_path,
            vjepa2_root=config.vjepa2_root,
            image_size=config.image_size,
            freeze=config.encoder_freeze,
        )

        # 2. Motion Branch (Load pretrained FlowTokenEncoder weights as adapter)
        self.motion_adapter = build_motion_encoder_adapter(
            checkpoint_path=config.motion_checkpoint_path,
            flow_channels=config.flow_channels,
            token_dim=config.flow_token_dim,
            freeze=config.motion_freeze,
        )
        # Convenience aliases used by checkpoints.py and forward()
        self.flow_encoder = self.motion_adapter.flow_encoder

        self.predictor = CausalPredictor(
            checkpoint_path=config.vitl_checkpoint_path,
            vjepa2_root=config.vjepa2_root,
            image_size=config.image_size,
            num_transitions=config.history_length - 1,
            latent_dim=config.latent_dim,
            confounder_dim=config.confounder_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            predictor_depth=config.predictor_depth,
            predictor_num_heads=config.predictor_num_heads,
            num_mask_tokens=config.predictor_num_mask_tokens,
            mode=config.mode,
            freeze=config.predictor_freeze,
        )

        self.fusion = ResidualFusion(config.latent_dim, config.flow_token_dim, config.fusion_scale)
        self.factorizer = LatentFactorizer(config.latent_dim)
        self.confounder = ConfounderGRU(config.latent_dim, config.confounder_dim)
        self.mask_sampler = MaskSampler(
            config.mask_ratio,
            image_size=config.image_size,
            vjepa2_root=config.vjepa2_root,
        )
        self.loss_fn = CaMoJEPALoss(
            lambda_jepa=config.jepa_loss_weight,
            lambda_orth=config.orthogonality_loss_weight,
            lambda_recon=config.reconstruction_loss_weight,
        )

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """EMA-update target encoder after an optimizer step on the context branch."""
        momentum_value = self.config.target_ema_momentum if momentum is None else momentum
        for target, source in zip(
            self.target_encoder.backbone.parameters(),
            self.context_encoder.backbone.parameters(),
        ):
            target.mul_(momentum_value).add_(source, alpha=1.0 - momentum_value)

    def forward(self, batch: FrameBatch) -> ModelOutput:
        batch.validate()

        # Step 1: Motion branch — [B, T-1, N_p, 1024]
        dynamic = self.motion_adapter(batch.images, flows=batch.flows)

        # Step 2: Static branch — keep full patch tokens [B, T-1, N_p, 1024]
        context_tokens = self.context_encoder.forward_tokens(batch.images)
        with torch.no_grad():
            target_tokens = self.target_encoder.forward_tokens(batch.images)

        # Step 3: Residual fusion (token-wise) — [B, T-1, N_p, 1024]
        fused = self.fusion(context_tokens, dynamic)

        # Step 4: Latent factorization (token-wise) — [B, T-1, N_p, 1024]
        z_task, z_exogenous = self.factorizer(fused)

        # Step 5: Confounder GRU — mean-pools patches internally → [B, confounder_dim]
        U = self.confounder(fused)

        # Step 6: Flatten spatiotemporal tokens for masking
        B, T_1, N_p, D = z_task.shape
        z_task_flat   = z_task.reshape(B, T_1 * N_p, D)
        target_flat   = target_tokens.reshape(B, T_1 * N_p, D)

        # Step 7: Spatiotemporal block masking
        z_task_masked, context_indices, mask_indices = self.mask_sampler(z_task_flat)

        # Step 8: Target patches at masked positions
        z_target = self._gather_tokens(target_flat, mask_indices)

        # Step 9: Causal predictor
        z_pred = self.predictor(z_task_masked, U, context_indices, mask_indices)

        return ModelOutput(
            z_pred=z_pred,
            target_patches=z_target,
            z_task=z_task_flat,
            z_exogenous=z_exogenous.reshape(B, T_1 * N_p, D),
            U=U,
            fused_features=fused,
            static_features=context_tokens,
            losses=None,
            mask_indices=mask_indices,
        )