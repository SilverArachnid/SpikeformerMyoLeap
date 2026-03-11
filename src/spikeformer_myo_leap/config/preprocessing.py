from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    dataset_root: str = "datasets"
    target_mode: str = "xyz"
    resample_hz: float = 100.0
    emg_window_size: int = 64
    use_wrist_relative_pose: bool = True
    validate_episodes: bool = True
