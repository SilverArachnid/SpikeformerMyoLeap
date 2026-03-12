"""Training and evaluation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from spikeformer_myo_leap.config import PreprocessingConfig


@dataclass
class DatasetConfig:
    """Dataset settings used to construct training windows.

    Attributes:
        preprocessing: Shared preprocessing settings. Defaults to ``PreprocessingConfig()``.
        dataset_root: Root dataset directory used when ``include_paths`` is empty. Defaults to ``"datasets"``.
        include_paths: Optional explicit list of dataset directories or episode folders to include.
            Each path may point to a dataset root, subject/session subtree, pose subtree, or a single
            ``ep_XXXX`` directory. Defaults to an empty list, which means "discover everything under
            ``dataset_root``".
        window_size: Number of EMG timesteps per training sample. Defaults to ``64``.
        stride: Step size between adjacent training targets. Defaults to ``1``.
    """

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    dataset_root: str = "datasets"
    include_paths: list[str] = field(default_factory=list)
    window_size: int = 64
    stride: int = 1


@dataclass
class SplitConfig:
    """Train/validation split settings."""

    train_fraction: float = 0.8
    seed: int = 42


@dataclass
class TrainingConfig:
    """Top-level training settings for packaged train entry points."""

    model_name: str = "spikeformer"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    num_epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    output_dir: str = "artifacts/train"
    device: str = "auto"
    num_workers: int = 0
    model_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Settings for checkpoint evaluation."""

    model_name: str = "spikeformer"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: int = 64
    checkpoint_path: str = ""
    device: str = "auto"
    num_workers: int = 0
    model_kwargs: dict[str, object] = field(default_factory=dict)
