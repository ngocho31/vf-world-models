"""Static and motion feature fusion."""

import torch
from torch import nn


class GatedCrossAttentionFusion(nn.Module):
    """Fuse static and dynamic tokens with zero-initialized cross-attention gating."""

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        fusion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if latent_dim % num_heads != 0:
            for head_count in (8, 4, 2, 1):
                if latent_dim % head_count == 0:
                    num_heads = head_count
                    break
        self.latent_dim = latent_dim
        self.fusion_scale = fusion_scale
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        if static.ndim not in (3, 4):
            raise ValueError(
                "GatedCrossAttentionFusion expects (B, N, D) or (B, T, N, D); "
                f"got {tuple(static.shape)}"
            )
        if static.shape[-1] != self.latent_dim:
            raise ValueError(
                f"static latent dimension must be {self.latent_dim}; got {static.shape[-1]}"
            )
        if dynamic.shape[-1] != self.latent_dim:
            raise ValueError(
                f"dynamic latent dimension must be {self.latent_dim}; got {dynamic.shape[-1]}"
            )

        if static.ndim == 4:
            batch, time, static_tokens, latent_dim = static.shape
            dynamic_tokens = dynamic.shape[2]
            query = static.reshape(batch * time, static_tokens, latent_dim)
            key_value = dynamic.reshape(batch * time, dynamic_tokens, latent_dim)
        else:
            query = static
            key_value = dynamic

        attn_out, _ = self.cross_attn(
            self.norm_q(query),
            self.norm_kv(key_value),
            self.norm_kv(key_value),
            need_weights=False,
        )
        fused = query + self.fusion_scale * torch.tanh(self.gate) * attn_out

        if static.ndim == 4:
            return fused.reshape(batch, time, static_tokens, latent_dim)
        return fused