"""Configuration types for dataset preprocessing."""

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Default settings for converting raw episodes into training-ready arrays.

    Attributes:
        dataset_root: Root directory containing recorded episodes. Defaults to ``"datasets"``.
        target_mode: Pose representation to load, either ``"xyz"`` or ``"xy"``. Defaults to ``"xyz"``.
        resample_hz: Target frequency for synchronized interpolation. Defaults to ``100.0``.
        emg_window_size: Planned temporal window size for downstream model datasets. Defaults to ``64``.
        use_wrist_relative_pose: Whether to normalize pose targets relative to the wrist. Defaults to ``True``.
        validate_episodes: Whether future preprocessing commands should run structural checks. Defaults to ``True``.
    """

    dataset_root: str = "datasets"
    target_mode: str = "xyz"
    resample_hz: float = 100.0
    emg_window_size: int = 64
    use_wrist_relative_pose: bool = True
    validate_episodes: bool = True
