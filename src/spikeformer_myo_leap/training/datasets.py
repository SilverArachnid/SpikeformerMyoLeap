"""Training dataset adapters built on top of the preprocessing layer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset

from spikeformer_myo_leap.config import PreprocessingConfig
from spikeformer_myo_leap.data.preprocessing import PreprocessedEpisode, preprocess_episode
from spikeformer_myo_leap.data.raw import EpisodePaths, list_episode_paths


@dataclass(frozen=True)
class WindowedSampleIndex:
    """Reference to a specific training sample inside one preprocessed episode."""

    episode_index: int
    target_index: int


class WindowedPoseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory windowed EMG-to-pose dataset built from preprocessed episodes."""

    def __init__(
        self,
        episode_paths: Sequence[EpisodePaths],
        preprocessing: PreprocessingConfig,
        window_size: int = 64,
        stride: int = 1,
    ) -> None:
        self.episode_paths = list(episode_paths)
        self.preprocessing = preprocessing
        self.window_size = window_size
        self.stride = stride

        self.episodes: list[PreprocessedEpisode] = [
            preprocess_episode(paths, preprocessing) for paths in self.episode_paths
        ]
        self.indices: list[WindowedSampleIndex] = self._build_indices()

    def _build_indices(self) -> list[WindowedSampleIndex]:
        """Build sample indices across all preprocessed episodes."""

        indices: list[WindowedSampleIndex] = []
        for episode_index, episode in enumerate(self.episodes):
            num_frames = episode.emg.shape[0]
            if num_frames <= self.window_size:
                continue
            for target_index in range(self.window_size, num_frames, self.stride):
                indices.append(WindowedSampleIndex(episode_index=episode_index, target_index=target_index))
        return indices

    def __len__(self) -> int:
        """Return the number of window-target pairs."""

        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one EMG window and its aligned pose target."""

        sample = self.indices[index]
        episode = self.episodes[sample.episode_index]
        start_index = sample.target_index - self.window_size
        emg_window = episode.emg[start_index : sample.target_index]
        pose_target = episode.pose[sample.target_index]
        return torch.from_numpy(emg_window), torch.from_numpy(pose_target)


def _normalize_include_paths(dataset_root: str, include_paths: Sequence[str]) -> list[str]:
    """Return absolute include paths relative to ``dataset_root`` when needed."""

    normalized: list[str] = []
    for path in include_paths:
        if os.path.isabs(path):
            normalized.append(path)
            continue
        candidate = os.path.join(dataset_root, path)
        normalized.append(candidate if os.path.exists(candidate) else path)
    return normalized


def collect_episode_paths(dataset_root: str, include_paths: Sequence[str] | None = None) -> list[EpisodePaths]:
    """Collect complete episode paths from ``dataset_root`` or an explicit include list.

    When ``include_paths`` is empty, all complete episodes under ``dataset_root`` are used.
    When ``include_paths`` is provided, each item may point to a higher-level subtree or directly
    to an ``ep_XXXX`` episode directory.
    """

    if not include_paths:
        return list_episode_paths(dataset_root)

    episodes_by_root: dict[str, EpisodePaths] = {}
    for include_path in _normalize_include_paths(dataset_root, include_paths):
        if os.path.isdir(include_path):
            nested_episodes = list_episode_paths(include_path)
            if nested_episodes:
                for episode in nested_episodes:
                    episodes_by_root[os.path.abspath(episode.root)] = episode
                continue

            filenames = set(os.listdir(include_path))
            if {"emg.csv", "pose.csv", "meta.json"}.issubset(filenames):
                episode = EpisodePaths(
                    root=include_path,
                    emg_csv=os.path.join(include_path, "emg.csv"),
                    pose_csv=os.path.join(include_path, "pose.csv"),
                    meta_json=os.path.join(include_path, "meta.json"),
                )
                episodes_by_root[os.path.abspath(episode.root)] = episode
                continue

        raise FileNotFoundError(
            f"Include path '{include_path}' does not resolve to any complete episodes."
        )

    return [episodes_by_root[root] for root in sorted(episodes_by_root)]


def build_windowed_dataset(
    dataset_root: str,
    preprocessing: PreprocessingConfig,
    include_paths: Sequence[str] | None = None,
    window_size: int = 64,
    stride: int = 1,
) -> WindowedPoseDataset:
    """Build a windowed dataset from all complete episodes under ``dataset_root``."""

    return WindowedPoseDataset(
        episode_paths=collect_episode_paths(dataset_root=dataset_root, include_paths=include_paths),
        preprocessing=preprocessing,
        window_size=window_size,
        stride=stride,
    )


def build_dataset_splits(
    dataset_root: str,
    preprocessing: PreprocessingConfig,
    include_paths: Sequence[str] | None = None,
    window_size: int = 64,
    stride: int = 1,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[WindowedPoseDataset, Subset[WindowedPoseDataset], Subset[WindowedPoseDataset]]:
    """Build train/validation splits from the windowed dataset."""

    dataset = build_windowed_dataset(
        dataset_root=dataset_root,
        preprocessing=preprocessing,
        include_paths=include_paths,
        window_size=window_size,
        stride=stride,
    )
    num_samples = len(dataset)
    train_size = int(num_samples * train_fraction)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    return dataset, Subset(dataset, train_indices), Subset(dataset, val_indices)
