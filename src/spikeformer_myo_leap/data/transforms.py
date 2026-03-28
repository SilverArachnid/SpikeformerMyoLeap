"""Pose-space transforms shared across preprocessing, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


POSE_JOINT_COUNT = 21

# Joint-angle targets skip the base-angle triples and focus on intra-finger articulation.
JOINT_ANGLE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
)


@dataclass(frozen=True)
class DatasetNormalizationStats:
    """Train-split normalization statistics reused across train/eval/inference.

    All arrays are optional because EMG normalization and target standardization are
    independently configurable.
    """

    emg_mean: list[float] | None = None
    emg_std: list[float] | None = None
    target_mean: list[float] | None = None
    target_std: list[float] | None = None

    def to_dict(self) -> dict[str, list[float] | None]:
        """Return a JSON-serializable representation."""

        return {
            "emg_mean": self.emg_mean,
            "emg_std": self.emg_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float] | None] | None) -> "DatasetNormalizationStats | None":
        """Build stats from serialized checkpoint data."""

        if not data:
            return None
        return cls(
            emg_mean=data.get("emg_mean"),
            emg_std=data.get("emg_std"),
            target_mean=data.get("target_mean"),
            target_std=data.get("target_std"),
        )

    def has_emg_stats(self) -> bool:
        """Return whether EMG standardization stats are available."""

        return self.emg_mean is not None and self.emg_std is not None

    def has_target_stats(self) -> bool:
        """Return whether target standardization stats are available."""

        return self.target_mean is not None and self.target_std is not None


def pose_axes(target_mode: str) -> int:
    """Return the number of coordinates stored per joint for a target mode."""

    if target_mode == "xy":
        return 2
    if target_mode == "xyz":
        return 3
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def target_feature_dim(target_mode: str, target_representation: str) -> int:
    """Return the flattened target dimension for a preprocessing configuration."""

    if target_representation == "points":
        return POSE_JOINT_COUNT * pose_axes(target_mode)
    if target_representation == "joint_angles":
        if target_mode != "xyz":
            raise ValueError("joint_angles target representation requires target_mode='xyz'.")
        return len(JOINT_ANGLE_TRIPLES)
    raise ValueError(f"Unsupported target_representation: {target_representation}")


def reshape_pose_values(
    pose_values: NDArray[np.float32],
    target_mode: str,
) -> NDArray[np.float32]:
    """Reshape flat pose features into ``[frames, joints, axes]`` form."""

    num_axes = pose_axes(target_mode)
    if pose_values.size == 0:
        return pose_values.reshape(0, POSE_JOINT_COUNT, num_axes).astype(np.float32)
    return pose_values.reshape(-1, POSE_JOINT_COUNT, num_axes).astype(np.float32)


def flatten_pose_values(pose_frames: NDArray[np.float32]) -> NDArray[np.float32]:
    """Flatten ``[frames, joints, axes]`` arrays back to ``[frames, features]``."""

    if pose_frames.size == 0:
        return pose_frames.reshape(0, 0).astype(np.float32)
    return pose_frames.reshape(pose_frames.shape[0], -1).astype(np.float32)


def _safe_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize vectors row-wise while keeping zero-length rows stable."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-6, 1.0, norms)
    return (vectors / safe_norms).astype(np.float32)


def make_wrist_relative_pose(
    pose_values: NDArray[np.float32],
    target_mode: str,
) -> NDArray[np.float32]:
    """Translate pose coordinates so the wrist becomes the per-frame origin."""

    num_axes = pose_axes(target_mode)
    if pose_values.size == 0:
        return pose_values.astype(np.float32)

    wrist = pose_values[:, :num_axes]
    tiled_wrist = np.tile(wrist, pose_values.shape[1] // num_axes)
    return (pose_values - tiled_wrist).astype(np.float32)


def make_palm_frame_pose(
    pose_values: NDArray[np.float32],
    target_mode: str,
) -> NDArray[np.float32]:
    """Rotate 3D pose coordinates into a palm-aligned local coordinate frame.

    The palm frame uses:
    - wrist as the origin
    - the index MCP and pinky MCP to define a palm span axis
    - the wrist-to-palm-center direction to derive the orthogonal in-plane axis

    This intentionally removes global hand orientation so the downstream model can
    focus on articulation rather than camera/forearm pose.
    """

    if target_mode != "xyz":
        raise ValueError("Palm-frame pose normalization requires target_mode='xyz'.")
    if pose_values.size == 0:
        return pose_values.astype(np.float32)

    frames = reshape_pose_values(pose_values, target_mode)
    wrist = frames[:, 0, :]
    index_mcp = frames[:, 5, :]
    pinky_mcp = frames[:, 17, :]

    span_axis = _safe_normalize(index_mcp - pinky_mcp)
    palm_center = (index_mcp + pinky_mcp) * 0.5
    palm_forward = _safe_normalize(palm_center - wrist)
    palm_normal = _safe_normalize(np.cross(span_axis, palm_forward))
    palm_up = _safe_normalize(np.cross(palm_normal, span_axis))

    relative = frames - wrist[:, None, :]
    basis = np.stack([span_axis, palm_up, palm_normal], axis=1)  # [frames, 3, 3]
    rotated = np.einsum("fjd,fad->fja", relative, basis).astype(np.float32)
    return flatten_pose_values(rotated)


def convert_pose_to_joint_angles(
    pose_values: NDArray[np.float32],
    target_mode: str,
) -> NDArray[np.float32]:
    """Convert 3D point targets into per-finger articulation angles in radians.

    Base angles are intentionally excluded. The returned feature order follows
    ``JOINT_ANGLE_TRIPLES`` and is suitable as a compact training target when
    only finger articulation matters.
    """

    if target_mode != "xyz":
        raise ValueError("joint_angles target representation requires target_mode='xyz'.")
    if pose_values.size == 0:
        return np.empty((0, len(JOINT_ANGLE_TRIPLES)), dtype=np.float32)

    frames = reshape_pose_values(pose_values, target_mode)
    angles: list[NDArray[np.float32]] = []
    for first_idx, center_idx, last_idx in JOINT_ANGLE_TRIPLES:
        first = frames[:, first_idx, :] - frames[:, center_idx, :]
        last = frames[:, last_idx, :] - frames[:, center_idx, :]
        first_unit = _safe_normalize(first)
        last_unit = _safe_normalize(last)
        cosine = np.sum(first_unit * last_unit, axis=1)
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(np.arccos(cosine).astype(np.float32))
    return np.stack(angles, axis=1).astype(np.float32)


def apply_standardization(
    values: NDArray[np.float32],
    mean: list[float] | None,
    std: list[float] | None,
) -> NDArray[np.float32]:
    """Apply feature standardization when statistics are available."""

    if mean is None or std is None or values.size == 0:
        return values.astype(np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    safe_std = np.where(std_arr < 1e-6, 1.0, std_arr)
    return ((values - mean_arr) / safe_std).astype(np.float32)


def fit_standardization(
    batches: list[NDArray[np.float32]],
) -> tuple[list[float] | None, list[float] | None]:
    """Fit feature-wise mean/std over a list of ``[time, features]`` arrays."""

    non_empty = [batch for batch in batches if batch.size > 0]
    if not non_empty:
        return None, None
    stacked = np.concatenate(non_empty, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()
