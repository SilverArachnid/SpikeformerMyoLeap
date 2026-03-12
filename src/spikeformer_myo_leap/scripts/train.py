"""Package entry point for model training."""

from __future__ import annotations

import argparse
import json
from typing import Any

from spikeformer_myo_leap.config import PreprocessingConfig
from spikeformer_myo_leap.models.registry import model_names
from spikeformer_myo_leap.training import DatasetConfig, SplitConfig, TrainingConfig, train_model


def _parse_model_kwargs(raw_value: str) -> dict[str, Any]:
    """Parse JSON model kwargs passed on the command line."""

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must decode to a JSON object.")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training CLI argument parser."""

    parser = argparse.ArgumentParser(description="Train an EMG-to-pose regression model.")
    parser.add_argument("--model-name", choices=model_names(), default="spikeformer", help="Model family to train.")
    parser.add_argument("--dataset-root", default="datasets", help="Dataset root directory to scan.")
    parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        dest="include_paths",
        help=(
            "Restrict training to a dataset subtree or specific episode. "
            "May be provided multiple times, e.g. --include-path user_1/session_2/test_pose"
        ),
    )
    parser.add_argument("--target-mode", choices=["xyz", "xy"], default="xyz", help="Pose target representation.")
    parser.add_argument("--resample-hz", type=float, default=100.0, help="Resampling frequency for preprocessing.")
    parser.add_argument("--window-size", type=int, default=64, help="Number of EMG timesteps per training sample.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between adjacent target frames.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples used for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--num-epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument("--output-dir", default="artifacts/train", help="Directory for training outputs.")
    parser.add_argument("--device", default="auto", help="Training device, e.g. auto, cpu, cuda.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help=(
            "JSON object of extra model constructor kwargs. "
            "Example: '{\"embed_dim\": 32, \"num_layers\": 2, \"heads\": 4}'"
        ),
    )
    return parser


def main() -> None:
    """Run model training from explicit CLI configuration."""

    parser = build_arg_parser()
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        dataset=DatasetConfig(
            dataset_root=args.dataset_root,
            include_paths=args.include_paths,
            preprocessing=PreprocessingConfig(
                dataset_root=args.dataset_root,
                target_mode=args.target_mode,
                resample_hz=args.resample_hz,
                emg_window_size=args.window_size,
            ),
            window_size=args.window_size,
            stride=args.stride,
        ),
        split=SplitConfig(train_fraction=args.train_fraction, seed=args.seed),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
        model_kwargs=_parse_model_kwargs(args.model_kwargs),
    )

    summary = train_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
