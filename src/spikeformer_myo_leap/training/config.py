"""Training and evaluation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
    full_episode_eval: dict[str, object] = field(
        default_factory=lambda: {
            "enabled": True,
            "every_n_epochs": 5,
            "num_episodes": 1,
            "save_visualizations": True,
        }
    )


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


def build_preprocessing_config(data: Mapping[str, Any]) -> PreprocessingConfig:
    """Construct ``PreprocessingConfig`` from a mapping-like config node."""

    return PreprocessingConfig(
        dataset_root=str(data.get("dataset_root", "datasets")),
        target_mode=str(data.get("target_mode", "xyz")),
        resample_hz=float(data.get("resample_hz", 100.0)),
        emg_window_size=int(data.get("emg_window_size", 64)),
        use_wrist_relative_pose=bool(data.get("use_wrist_relative_pose", True)),
        validate_episodes=bool(data.get("validate_episodes", True)),
    )


def build_dataset_config(data: Mapping[str, Any]) -> DatasetConfig:
    """Construct ``DatasetConfig`` from a mapping-like config node."""

    preprocessing_node = data.get("preprocessing", {})
    include_paths = data.get("include_paths", [])
    return DatasetConfig(
        preprocessing=build_preprocessing_config(preprocessing_node),
        dataset_root=str(data.get("dataset_root", "datasets")),
        include_paths=[str(path) for path in include_paths],
        window_size=int(data.get("window_size", 64)),
        stride=int(data.get("stride", 1)),
    )


def build_split_config(data: Mapping[str, Any]) -> SplitConfig:
    """Construct ``SplitConfig`` from a mapping-like config node."""

    return SplitConfig(
        train_fraction=float(data.get("train_fraction", 0.8)),
        seed=int(data.get("seed", 42)),
    )


def build_training_config(data: Mapping[str, Any]) -> TrainingConfig:
    """Construct ``TrainingConfig`` from a Hydra/OmegaConf mapping."""

    model_node = data.get("model", {})
    return TrainingConfig(
        model_name=str(model_node.get("model_name", data.get("model_name", "spikeformer"))),
        dataset=build_dataset_config(data.get("dataset", {})),
        split=build_split_config(data.get("split", {})),
        num_epochs=int(data.get("num_epochs", 8)),
        batch_size=int(data.get("batch_size", 64)),
        learning_rate=float(data.get("learning_rate", 1e-4)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        output_dir=str(data.get("output_dir", "artifacts/train")),
        device=str(data.get("device", "auto")),
        num_workers=int(data.get("num_workers", 0)),
        model_kwargs=dict(model_node.get("model_kwargs", data.get("model_kwargs", {}))),
        full_episode_eval=dict(data.get("full_episode_eval", {})),
    )


def build_evaluation_config(data: Mapping[str, Any]) -> EvaluationConfig:
    """Construct ``EvaluationConfig`` from a Hydra/OmegaConf mapping."""

    model_node = data.get("model", {})
    return EvaluationConfig(
        model_name=str(model_node.get("model_name", data.get("model_name", "spikeformer"))),
        dataset=build_dataset_config(data.get("dataset", {})),
        batch_size=int(data.get("batch_size", 64)),
        checkpoint_path=str(data.get("checkpoint_path", "")),
        device=str(data.get("device", "auto")),
        num_workers=int(data.get("num_workers", 0)),
        model_kwargs=dict(model_node.get("model_kwargs", data.get("model_kwargs", {}))),
    )
