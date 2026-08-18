"""PyTorch data loading for the common ``dataset_camo`` episode format."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import CaMoJEPAConfig
from ..contracts import FrameBatch


@dataclass(frozen=True)
class EpisodeWindow:
    dataset_root: Path
    episode_path: Path
    start_index: int


class CaMoEpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    """Load fixed-length image/ego-motion windows from converted CaMo episodes.

    ``dataset_root`` can identify one converted source:

    .. code-block:: text

        dataset_camo/navsim/
          episodes/<split>/<log_name>.npz
          images/<split>/<log_name>/<token>.jpg

    Or it can identify the aggregate ``dataset_camo`` directory, whose direct
    child directories with an ``episodes/`` directory are discovered as sources.
    Each episode must contain ``timestamps_us``, ``image_paths``, ``can_bus``,
    and ``ego_motion`` arrays with the same leading frame dimension. Image paths
    are relative to the source dataset root.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        history_length: int = 4,
        stride: int = 1,
        image_size: int | tuple[int, int] = (512, 256),
        max_cached_episodes: int = 8,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or any(size < 1 for size in image_size):
            raise ValueError("image_size must be a positive (width, height) pair")
        if max_cached_episodes < 1:
            raise ValueError("max_cached_episodes must be positive")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.history_length = history_length
        self.stride = stride
        self.image_size = image_size
        self.max_cached_episodes = max_cached_episodes
        self._episode_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._windows = self._discover_windows()
        if not self._windows:
            raise ValueError(
                f"no episodes in {self.dataset_root} for split={split!r} "
                f"with history_length={history_length}"
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self._windows[index]
        episode = self._load_episode(window.episode_path)
        stop_index = window.start_index + self.history_length
        image_paths = episode["image_paths"][window.start_index:stop_index]
        images = torch.stack(
            [self._load_image(window.dataset_root / str(path)) for path in image_paths]
        )
        ego_motion = torch.from_numpy(
            episode["ego_motion"][window.start_index:stop_index].astype(np.float32, copy=False)
        )
        can_bus = torch.from_numpy(
            episode["can_bus"][window.start_index:stop_index].astype(np.float32, copy=False)
        )
        timestamps_us = torch.from_numpy(
            episode["timestamps_us"][window.start_index:stop_index].astype(np.int64, copy=False)
        )
        return {
            "images": images,
            "ego_motion": ego_motion,
            "can_bus": can_bus,
            "timestamps_us": timestamps_us,
        }

    def _discover_windows(self) -> list[EpisodeWindow]:
        windows: list[EpisodeWindow] = []
        for source_root in self._source_roots():
            for episode_path in sorted((source_root / "episodes" / self.split).glob("*.npz")):
                episode = self._load_episode(episode_path)
                frame_count = len(episode["timestamps_us"])
                for start_index in range(0, frame_count - self.history_length + 1, self.stride):
                    windows.append(EpisodeWindow(source_root, episode_path, start_index))
        return windows

    def _source_roots(self) -> list[Path]:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"CaMo dataset root does not exist: {self.dataset_root}")
        if (self.dataset_root / "episodes").is_dir():
            return [self.dataset_root]
        source_roots = sorted(
            path for path in self.dataset_root.iterdir() if path.is_dir() and (path / "episodes").is_dir()
        )
        if not source_roots:
            raise FileNotFoundError(
                f"expected episodes/ in {self.dataset_root} or one of its immediate child directories"
            )
        return source_roots

    def _load_episode(self, episode_path: Path) -> dict[str, np.ndarray]:
        cached = self._episode_cache.get(episode_path)
        if cached is not None:
            self._episode_cache.move_to_end(episode_path)
            return cached

        with np.load(episode_path, allow_pickle=False) as raw_episode:
            required = {"timestamps_us", "image_paths", "can_bus", "ego_motion"}
            missing = required.difference(raw_episode.files)
            if missing:
                raise ValueError(f"invalid CaMo episode {episode_path}: missing {sorted(missing)}")
            episode = {name: raw_episode[name] for name in required}

        frame_count = len(episode["timestamps_us"])
        if any(len(episode[name]) != frame_count for name in ("image_paths", "can_bus", "ego_motion")):
            raise ValueError(f"invalid CaMo episode {episode_path}: inconsistent frame dimensions")
        if episode["ego_motion"].ndim != 2 or episode["can_bus"].ndim != 2:
            raise ValueError(f"invalid CaMo episode {episode_path}: ego_motion and can_bus must be matrices")

        self._episode_cache[episode_path] = episode
        if len(self._episode_cache) > self.max_cached_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    def _load_image(self, image_path: Path) -> torch.Tensor:
        if not image_path.is_file():
            raise FileNotFoundError(f"CaMo image does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def collate_camo_samples(samples: Sequence[dict[str, torch.Tensor]]) -> FrameBatch:
    """Collate CaMo dataset samples into the pipeline's ``FrameBatch`` contract."""
    if not samples:
        raise ValueError("cannot collate an empty list of CaMo samples")
    batch = FrameBatch(
        images=torch.stack([sample["images"] for sample in samples]),
        ego_motion=torch.stack([sample["ego_motion"] for sample in samples]),
    )
    batch.validate()
    return batch


def make_camo_dataloader(
    config: CaMoJEPAConfig,
) -> DataLoader[FrameBatch]:
    """Create the configured DataLoader yielding ``FrameBatch(images, ego_motion)``."""
    dataset = CaMoEpisodeDataset(
        dataset_root=config.dataset_root,
        split=config.dataset_split,
        history_length=config.history_length,
        stride=config.stride,
        image_size=config.image_size,
        max_cached_episodes=config.max_cached_episodes,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_camo_samples,
    )