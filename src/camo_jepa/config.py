"""Configuration objects for CaMo-JEPA training."""

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
    max_cached_episodes: int = 16
    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 4

    # Trainable model components
    freeze_context_encoder: bool = True
    freeze_target_encoder: bool = True
    freeze_flow_estimator: bool = True
    freeze_motion_encoder: bool = False
    freeze_fusion: bool = False
    freeze_factorizer: bool = False
    freeze_confounder: bool = False
    freeze_predictor: bool = False

    # Ablation study flags
    ablation_motion_branch: bool = False # True: disable the motion branch
    ablation_confounder: bool = False # True: disable the confounder branch
    ablation_factorizer: bool = False # True: disable the factorizer branch

    # Drive-JEPA ViT-L checkpoint and V-JEPA2 root path for loading pretrained weights
    vitl_checkpoint_path: str = ".cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"
    vjepa2_root: str = "src/vjepa2"

    # FlowFormer++ checkpoint and root path
    motion_estimator_checkpoint_path: str = ".cache/checkpoints/motion_estimator/kitti_finetune_vf.pth"
    flowformer_root: str = "src/flowformer"

    # Motion encoder parameters
    flow_channels: int = 2

    # Fusion parameters
    fusion_num_heads: int = 16
    fusion_scale: float = 1.0

    # Latent Factorizer parameters
    task_dim: int = 512
    enforce_orthogonality: bool = True

    # Confounder parameters
    confounder_dim: int = 128

    # Predictor parameters
    predictor_embed_dim: int = 384
    predictor_depth: int = 12
    predictor_num_heads: int = 12
    predictor_num_mask_tokens: int = 10
    predictor_confounder_mode: str = "add"

    # Masking parameters
    mask_ratio: float = 0.7

    # CaMo-JEPA training parameters
    output_checkpoint_path: str = "outputs/camo-jepa/checkpoints/camo.pt"
    output_log_dir: str = "outputs/camo-jepa/logs"
    pretrained: bool = True
    latent_dim: int = 1024
    num_epochs: int = 50
    n_steps_per_epoch: int = 100
    learning_rate: float = 0.000525
    target_ema_momentum: float = 0.99925
    # Loss weights
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
        if self.fusion_num_heads < 1:
            raise ValueError("fusion_num_heads must be positive")
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
        if not self.output_checkpoint_path:
            raise ValueError("output_checkpoint_path must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
