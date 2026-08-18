"""Public API for the CaMo-JEPA pipeline and training framework."""

from .config import CaMoJEPAConfig
from .contracts import FrameBatch, ModelOutput
from .pipeline import CaMoJEPAPipeline, load_checkpoint, save_checkpoint, train_step

__all__ = [
    "CaMoJEPAConfig",
    "FrameBatch",
    "ModelOutput",
    "CaMoJEPAPipeline",
    "train_step",
    "save_checkpoint",
    "load_checkpoint",
]