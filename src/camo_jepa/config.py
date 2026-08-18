"""Configuration objects shared by independently runnable pipeline stages."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CaMoJEPAConfig:
    # Dataset and dataloader parameters
    dataset_root: str = "dataset_camo/navsim"
    dataset_split: str = "trainval"
    history_length: int = 16
    stride: int = 1
    image_size: Tuple[int, int] = (512, 256)
    max_cached_episodes: int = 8
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 0

    stage: str = "camo_jepa"  # "motion_pretrain" or "camo_jepa"

    # Drive-JEPA ViT-L checkpoint and V-JEPA2 root path for loading pretrained weights
    vitl_checkpoint_path: str = ".cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"
    vjepa2_root: str = "vf-drive-jepa/vjepa2"
    encoder_freeze: bool = True

    # Motion encoder parameters
    motion_checkpoint_path: str = ".cache/checkpoints/motion_encoder/motion_encoder.pt"
    motion_freeze: bool = True

    # Drive-JEPA predictor parameters
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_num_heads: int = 12

    # CaMo-JEPA training parameters
    latent_dim: int = 1024
    confounder_dim: int = 128
    learning_rate: float = 0.000525
    predictor_num_mask_tokens: int = 10
    mode: str = "add"
    predictor_freeze: bool = False
    predictor_output_checkpoint_path: str = "outputs/camo-jepa/checkpoints/predictor.pt"
    flow_channels: int = 2
    flow_token_dim: int = 1024
    mask_ratio: float = 0.7
    fusion_scale: float = 1.0
    target_ema_momentum: float = 0.99925
    # Loss weights for the CaMo-JEPA objective
    jepa_loss_weight: float = 1.0
    orthogonality_loss_weight: float = 0.01
    reconstruction_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.history_length < 2:
            raise ValueError("history_length must be at least 2 for consecutive frame clips")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if len(self.image_size) != 2 or any(size < 1 for size in self.image_size):
            raise ValueError("image_size must be a positive (width, height) pair")
        if self.latent_dim != 1024:
            raise ValueError("latent_dim must be 1024 to match the Drive-JEPA ViT-L encoder")
        if self.flow_token_dim != self.latent_dim:
            raise ValueError("flow_token_dim must match latent_dim for residual fusion")
        if not 0.0 < self.target_ema_momentum <= 1.0:
            raise ValueError("target_ema_momentum must be in (0, 1]")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.predictor_embed_dim < 1 or self.predictor_depth < 1 or self.predictor_num_heads < 1:
            raise ValueError("predictor dimensions must be positive")
        if self.predictor_num_mask_tokens < 1:
            raise ValueError("predictor_num_mask_tokens must be positive")
        if not self.dataset_root:
            raise ValueError("dataset_root must not be empty")
        if not self.dataset_split:
            raise ValueError("dataset_split must not be empty")
        if not self.vitl_checkpoint_path:
            raise ValueError("vitl_checkpoint_path must not be empty")
        if not self.vjepa2_root:
            raise ValueError("vjepa2_root must not be empty")
        if not self.predictor_output_checkpoint_path:
            raise ValueError("predictor_output_checkpoint_path must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
