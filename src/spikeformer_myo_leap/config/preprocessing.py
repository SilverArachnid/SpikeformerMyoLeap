"""Configuration types for dataset preprocessing."""

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Default settings for converting raw episodes into training-ready arrays.

    Attributes:
        dataset_root: Root directory containing recorded episodes. Defaults to ``"datasets"``.
        target_mode: Pose coordinate mode to load from disk, either ``"xyz"`` or ``"xy"``.
            Defaults to ``"xyz"``.
        target_representation: Model target representation. ``"points"`` keeps joint coordinates,
            while ``"joint_angles"`` converts the palm-normalized 3D pose into per-finger joint
            angles. Defaults to ``"points"``.
        resample_hz: Target frequency for synchronized interpolation. Defaults to ``100.0``.
        emg_window_size: Planned temporal window size for downstream model datasets. Defaults to ``64``.
        use_wrist_relative_pose: Whether to normalize pose targets relative to the wrist. Defaults to ``True``.
        use_palm_frame_pose: Whether to rotate 3D poses into a palm-aligned local coordinate frame
            using the wrist, index MCP, and pinky MCP markers. Defaults to ``True``.
        normalize_emg: Whether train/eval pipelines should standardize EMG per channel using train-split
            statistics. Defaults to ``True``.
        standardize_targets: Whether model targets should be standardized using train-split statistics.
            Defaults to ``True``.
        validate_episodes: Whether future preprocessing commands should run structural checks. Defaults to ``True``.
    """

    dataset_root: str = "datasets"
    target_mode: str = "xyz"
    target_representation: str = "points"
    resample_hz: float = 100.0
    emg_window_size: int = 64
    use_wrist_relative_pose: bool = True
    use_palm_frame_pose: bool = True
    normalize_emg: bool = True
    standardize_targets: bool = True
    validate_episodes: bool = True
