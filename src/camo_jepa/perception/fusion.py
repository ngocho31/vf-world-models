"""Static and motion feature fusion."""

import torch
from torch import nn


class ResidualFusion(nn.Module):
    def __init__(self, latent_dim: int, dynamic_dim: int | None = None, fusion_scale: float = 1.0) -> None:
        super().__init__()
        self.dynamic_projection = (
            nn.Identity() if dynamic_dim in (None, latent_dim) else nn.Linear(dynamic_dim, latent_dim)
        )
        self.gate = nn.Parameter(torch.zeros(latent_dim))
        self.fusion_scale = fusion_scale

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        if static.shape[:-1] != dynamic.shape[:-1]:
            raise ValueError("static and dynamic latents must share batch and time dimensions")
        dynamic = self.dynamic_projection(dynamic)
        return static + self.fusion_scale * torch.tanh(self.gate) * dynamic
    
class GatedCrossAttentionFusion(nn.Module):
    """Zero-initialized gated cross-attention fusion.

        z_t = e_static + tanh(g) * CrossAttention(q=e_static, kv=e_dynamic)

    ``g`` is initialized to 0 so at the start of training ``z_t == e_static``
    (backward-compatible with frozen ViT). Supports:

      - (B, N, D)
      - (B, T, N, D)  — attention applied per time step (reshape to B*T)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dynamic_dim: int | None = None,
        fusion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            # fall back to a valid head count
            for h in (8, 4, 2, 1):
                if embed_dim % h == 0:
                    num_heads = h
                    break
        self.embed_dim = embed_dim
        self.fusion_scale = fusion_scale
        self.dynamic_projection = (
            nn.Identity() if dynamic_dim in (None, embed_dim) else nn.Linear(dynamic_dim, embed_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # scalar g, init 0
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, e_static: torch.Tensor, e_dynamic: torch.Tensor) -> torch.Tensor:
        e_dynamic = self.dynamic_projection(e_dynamic)

        # Flatten time if present: (B, T, N, D) -> (B*T, N, D)
        leading = e_static.shape[:-2]
        if e_static.ndim == 4:
            B, T, N, D = e_static.shape
            if e_dynamic.shape[:2] != (B, T) or e_dynamic.shape[-1] != D:
                raise ValueError(
                    f"static/dynamic shape mismatch: {tuple(e_static.shape)} vs {tuple(e_dynamic.shape)}"
                )
            q = e_static.reshape(B * T, N, D)
            # dynamic may have different N (flow resolution); keep its N
            Nd = e_dynamic.shape[2]
            kv = e_dynamic.reshape(B * T, Nd, D)
        elif e_static.ndim == 3:
            q = e_static
            kv = e_dynamic
            B = T = None
            N = e_static.shape[1]
            D = e_static.shape[2]
        else:
            raise ValueError(
                f"GatedCrossAttentionFusion expects (B,N,D) or (B,T,N,D); got {tuple(e_static.shape)}"
            )

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(qn, kvn, kvn, need_weights=False)
        z = q + self.fusion_scale * torch.tanh(self.gate) * attn_out

        if e_static.ndim == 4:
            z = z.reshape(B, T, N, D)
        return z