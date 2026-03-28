"""Training dataset adapters built on top of the preprocessing layer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Sequence

import torch
from torch.utils.data import Dataset

from spikeformer_myo_leap.config import PreprocessingConfig
from spikeformer_myo_leap.data import (
    DatasetNormalizationStats,
    EpisodePaths,
    PreprocessedEpisode,
    apply_standardization,
    fit_standardization,
    list_episode_paths,
    preprocess_episode,
)


@dataclass(frozen=True)
class WindowedSampleIndex:
    """Reference to a specific training sample inside one preprocessed episode."""

    episode_index: int
    target_index: int


class WindowedPoseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory windowed EMG-to-target dataset built from preprocessed episodes."""

    def __init__(
        self,
        episode_paths: Sequence[EpisodePaths] | None,
        preprocessing: PreprocessingConfig,
        window_size: int = 64,
        stride: int = 1,
        episodes: Sequence[PreprocessedEpisode] | None = None,
        normalization_stats: DatasetNormalizationStats | None = None,
    ) -> None:
        self.episode_paths = list(episode_paths or [])
        self.preprocessing = preprocessing
        self.window_size = window_size
        self.stride = stride
        self.normalization_stats = normalization_stats

        base_episodes = (
            list(episodes)
            if episodes is not None
            else [preprocess_episode(paths, preprocessing) for paths in self.episode_paths]
        )
        self.episodes: list[PreprocessedEpisode] = apply_dataset_normalization(base_episodes, normalization_stats)
        self.indices: list[WindowedSampleIndex] = self._build_indices()
        self.target_dim = self.episodes[0].pose.shape[1] if self.episodes else 0

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
        """Return one EMG window and its aligned target."""

        sample = self.indices[index]
        episode = self.episodes[sample.episode_index]
        start_index = sample.target_index - self.window_size
        emg_window = episode.emg[start_index:sample.target_index]
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
    """Collect complete episode paths from ``dataset_root`` or an explicit include list."""

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


def preprocess_episodes(
    episode_paths: Sequence[EpisodePaths],
    preprocessing: PreprocessingConfig,
) -> list[PreprocessedEpisode]:
    """Preprocess raw episode paths into aligned training-ready arrays."""

    return [preprocess_episode(paths, preprocessing) for paths in episode_paths]


def fit_dataset_normalization_stats(
    episodes: Sequence[PreprocessedEpisode],
    preprocessing: PreprocessingConfig,
) -> DatasetNormalizationStats | None:
    """Fit train-split standardization stats for EMG and model targets."""

    use_emg = preprocessing.normalize_emg
    use_targets = preprocessing.standardize_targets
    if not use_emg and not use_targets:
        return None

    emg_mean, emg_std = fit_standardization([episode.emg for episode in episodes]) if use_emg else (None, None)
    target_mean, target_std = (
        fit_standardization([episode.pose for episode in episodes]) if use_targets else (None, None)
    )
    return DatasetNormalizationStats(
        emg_mean=emg_mean,
        emg_std=emg_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def apply_dataset_normalization(
    episodes: Sequence[PreprocessedEpisode],
    normalization_stats: DatasetNormalizationStats | None,
) -> list[PreprocessedEpisode]:
    """Apply fitted standardization stats to preprocessed episodes."""

    if normalization_stats is None:
        return list(episodes)

    normalized: list[PreprocessedEpisode] = []
    for episode in episodes:
        normalized.append(
            replace(
                episode,
                emg=apply_standardization(
                    episode.emg,
                    normalization_stats.emg_mean,
                    normalization_stats.emg_std,
                ),
                pose=apply_standardization(
                    episode.pose,
                    normalization_stats.target_mean,
                    normalization_stats.target_std,
                ),
            )
        )
    return normalized


def build_windowed_dataset(
    dataset_root: str,
    preprocessing: PreprocessingConfig,
    include_paths: Sequence[str] | None = None,
    window_size: int = 64,
    stride: int = 1,
    normalization_stats: DatasetNormalizationStats | None = None,
) -> WindowedPoseDataset:
    """Build a windowed dataset from all complete episodes under ``dataset_root``."""

    episode_paths = collect_episode_paths(dataset_root=dataset_root, include_paths=include_paths)
    episodes = preprocess_episodes(episode_paths, preprocessing)
    return WindowedPoseDataset(
        episode_paths=episode_paths,
        preprocessing=preprocessing,
        window_size=window_size,
        stride=stride,
        episodes=episodes,
        normalization_stats=normalization_stats,
    )


def build_dataset_splits(
    dataset_root: str,
    preprocessing: PreprocessingConfig,
    include_paths: Sequence[str] | None = None,
    window_size: int = 64,
    stride: int = 1,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[WindowedPoseDataset, WindowedPoseDataset, WindowedPoseDataset, DatasetNormalizationStats | None]:
    """Build episode-disjoint train/validation splits from the windowed dataset.

    The split happens at the episode level first, then train-split normalization
    stats are fitted and applied consistently to train/val/full datasets.
    """

    episode_paths = collect_episode_paths(dataset_root=dataset_root, include_paths=include_paths)
    episodes = preprocess_episodes(episode_paths, preprocessing)
    num_episodes = len(episode_paths)
    train_size = int(num_episodes * train_fraction)
    train_size = min(max(train_size, 1), max(num_episodes - 1, 1)) if num_episodes > 1 else num_episodes
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_episodes, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_episode_paths = [episode_paths[index] for index in train_indices]
    val_episode_paths = [episode_paths[index] for index in val_indices]
    train_episodes = [episodes[index] for index in train_indices]
    val_episodes = [episodes[index] for index in val_indices]

    normalization_stats = fit_dataset_normalization_stats(train_episodes, preprocessing)
    full_dataset = WindowedPoseDataset(
        episode_paths=episode_paths,
        preprocessing=preprocessing,
        window_size=window_size,
        stride=stride,
        episodes=episodes,
        normalization_stats=normalization_stats,
    )
    train_dataset = WindowedPoseDataset(
        episode_paths=train_episode_paths,
        preprocessing=preprocessing,
        window_size=window_size,
        stride=stride,
        episodes=train_episodes,
        normalization_stats=normalization_stats,
    )
    val_dataset = WindowedPoseDataset(
        episode_paths=val_episode_paths,
        preprocessing=preprocessing,
        window_size=window_size,
        stride=stride,
        episodes=val_episodes,
        normalization_stats=normalization_stats,
    )
    return full_dataset, train_dataset, val_dataset, normalization_stats
