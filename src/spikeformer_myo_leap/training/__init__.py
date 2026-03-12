"""Training and evaluation package."""

from .config import DatasetConfig, EvaluationConfig, SplitConfig, TrainingConfig
from .datasets import WindowedPoseDataset, build_dataset_splits
from .evaluate import evaluate_model
from .train import train_model

__all__ = [
    "DatasetConfig",
    "EvaluationConfig",
    "SplitConfig",
    "TrainingConfig",
    "WindowedPoseDataset",
    "build_dataset_splits",
    "evaluate_model",
    "train_model",
]
