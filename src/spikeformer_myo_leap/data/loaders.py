"""Array-oriented loaders built on top of raw episode CSV files."""

import numpy as np
from numpy.typing import NDArray

from .contracts import LANDMARK_NAMES
from .raw import EpisodePaths, load_episode


def emg_columns() -> list[str]:
    """Return the canonical eight EMG channel column names."""

    return [f"Channel_{index}" for index in range(1, 9)]


def pose_columns(target_mode: str = "xyz") -> list[str]:
    """Return pose CSV columns for the requested target representation."""

    if target_mode not in {"xy", "xyz"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    axes = ["X", "Y"] if target_mode == "xy" else ["X", "Y", "Z"]
    return [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in axes]


def load_emg_array(episode_paths: EpisodePaths) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Load one episode's EMG timestamps and channel values as NumPy arrays."""

    episode = load_episode(episode_paths)
    frame = episode["emg"]
    return frame["Timestamp_ms"].to_numpy(dtype=np.float32), frame[emg_columns()].to_numpy(dtype=np.float32)


def load_pose_array(
    episode_paths: EpisodePaths,
    target_mode: str = "xyz",
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Load one episode's pose timestamps and target values as NumPy arrays."""

    episode = load_episode(episode_paths)
    frame = episode["pose"]
    return frame["Timestamp_ms"].to_numpy(dtype=np.float32), frame[pose_columns(target_mode)].to_numpy(dtype=np.float32)
