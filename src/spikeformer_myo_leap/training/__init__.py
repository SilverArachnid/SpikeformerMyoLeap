"""Training and evaluation package."""

from .config import (
    DatasetConfig,
    EvaluationConfig,
    SplitConfig,
    TrainingConfig,
    build_dataset_config,
    build_evaluation_config,
    build_preprocessing_config,
    build_split_config,
    build_training_config,
)
from .datasets import WindowedPoseDataset, build_dataset_splits
from .evaluate import evaluate_model
from .full_episode import FullEpisodeValidationResult, run_full_episode_validation
from .train import train_model

__all__ = [
    "DatasetConfig",
    "EvaluationConfig",
    "SplitConfig",
    "TrainingConfig",
    "build_dataset_config",
    "build_evaluation_config",
    "build_preprocessing_config",
    "build_split_config",
    "build_training_config",
    "FullEpisodeValidationResult",
    "WindowedPoseDataset",
    "build_dataset_splits",
    "evaluate_model",
    "run_full_episode_validation",
    "train_model",
]
