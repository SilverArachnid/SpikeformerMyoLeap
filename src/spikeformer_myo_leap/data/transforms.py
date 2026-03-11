"""Pose-space transforms shared across preprocessing and future runtime inference."""

import numpy as np
from numpy.typing import NDArray


def pose_axes(target_mode: str) -> int:
    """Return the number of coordinates stored per joint for a target mode."""

    if target_mode == "xy":
        return 2
    if target_mode == "xyz":
        return 3
    raise ValueError(f"Unsupported target_mode: {target_mode}")


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
