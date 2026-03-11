import numpy as np


def pose_axes(target_mode: str) -> int:
    if target_mode == "xy":
        return 2
    if target_mode == "xyz":
        return 3
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def make_wrist_relative_pose(pose_values: np.ndarray, target_mode: str) -> np.ndarray:
    num_axes = pose_axes(target_mode)
    if pose_values.size == 0:
        return pose_values.astype(np.float32)

    wrist = pose_values[:, :num_axes]
    tiled_wrist = np.tile(wrist, pose_values.shape[1] // num_axes)
    return (pose_values - tiled_wrist).astype(np.float32)
