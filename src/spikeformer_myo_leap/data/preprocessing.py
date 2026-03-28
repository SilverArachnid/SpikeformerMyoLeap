"""Preprocessing utilities for resampling and normalizing recorded episodes."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spikeformer_myo_leap.config import PreprocessingConfig

from .loaders import load_emg_array, load_pose_array
from .raw import EpisodePaths, load_episode_metadata
from .transforms import convert_pose_to_joint_angles, make_palm_frame_pose, make_wrist_relative_pose


@dataclass
class PreprocessedEpisode:
    """Container for one preprocessed episode aligned to a shared timebase."""

    episode_dir: str
    emg_timestamps_ms: NDArray[np.float32]
    emg: NDArray[np.float32]
    pose_timestamps_ms: NDArray[np.float32]
    pose: NDArray[np.float32]
    target_mode: str
    target_representation: str
    metadata: dict[str, Any]


def _resample_series(
    timestamps_ms: NDArray[np.float32],
    values: NDArray[np.float32],
    target_timestamps_ms: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Interpolate a multi-channel timeseries onto a target timestamp grid."""

    if len(timestamps_ms) == 0 or len(values) == 0:
        return np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=np.float32)

    if len(timestamps_ms) == 1:
        repeated = np.repeat(values[:1], len(target_timestamps_ms), axis=0)
        return repeated.astype(np.float32)

    resampled_columns = [
        np.interp(target_timestamps_ms, timestamps_ms, values[:, column_index])
        for column_index in range(values.shape[1])
    ]
    return np.stack(resampled_columns, axis=1).astype(np.float32)


def build_target_timestamps(duration_seconds: float, resample_hz: float) -> NDArray[np.float32]:
    """Construct the synchronized timestamp grid used for one episode."""

    if duration_seconds <= 0.0 or resample_hz <= 0.0:
        return np.empty((0,), dtype=np.float32)
    num_steps = max(int(round(duration_seconds * resample_hz)), 1)
    return np.linspace(0.0, duration_seconds * 1000.0, num=num_steps, endpoint=False, dtype=np.float32)


def preprocess_episode(
    episode_paths: EpisodePaths,
    config: PreprocessingConfig,
) -> PreprocessedEpisode:
    """Load, resample, and normalize a single episode according to ``config``."""

    metadata = load_episode_metadata(episode_paths.meta_json)
    duration_seconds = float(metadata.get("recorded_duration_seconds", 0.0))
    target_timestamps_ms = build_target_timestamps(duration_seconds, config.resample_hz)

    emg_timestamps_ms, emg_values = load_emg_array(episode_paths)
    pose_timestamps_ms, pose_values = load_pose_array(episode_paths, target_mode=config.target_mode)

    resampled_emg = _resample_series(emg_timestamps_ms, emg_values, target_timestamps_ms)
    resampled_pose = _resample_series(pose_timestamps_ms, pose_values, target_timestamps_ms)

    if config.use_wrist_relative_pose and len(resampled_pose) > 0:
        resampled_pose = make_wrist_relative_pose(resampled_pose, config.target_mode)
    if config.use_palm_frame_pose and len(resampled_pose) > 0:
        resampled_pose = make_palm_frame_pose(resampled_pose, config.target_mode)
    if config.target_representation == "joint_angles":
        resampled_pose = convert_pose_to_joint_angles(resampled_pose, config.target_mode)

    return PreprocessedEpisode(
        episode_dir=episode_paths.root,
        emg_timestamps_ms=target_timestamps_ms,
        emg=resampled_emg,
        pose_timestamps_ms=target_timestamps_ms,
        pose=resampled_pose,
        target_mode=config.target_mode,
        target_representation=config.target_representation,
        metadata=metadata,
    )
