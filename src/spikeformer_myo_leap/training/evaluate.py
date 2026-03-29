"""Reusable checkpoint evaluation logic.

Standalone evaluation reuses the train-split normalization statistics saved in
the checkpoint. Metrics are reported in the original target space by inverting
target standardization before computing RMSE/MAE.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikeformer_myo_leap.data import DatasetNormalizationStats, invert_standardization
from spikeformer_myo_leap.models import create_model

from .config import EvaluationConfig
from .datasets import build_windowed_dataset
from .train import reset_model_state, resolve_device


def evaluate_model(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate a saved checkpoint against the current dataset config."""

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set before running evaluation.")

    start_time = time.perf_counter()
    device = resolve_device(config.device)
    checkpoint_payload = torch.load(config.checkpoint_path, map_location=device)
    if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["model_state_dict"]
        normalization_stats = DatasetNormalizationStats.from_dict(checkpoint_payload.get("normalization_stats"))
    else:
        state_dict = checkpoint_payload
        normalization_stats = None
    dataset = build_windowed_dataset(
        dataset_root=config.dataset.dataset_root,
        preprocessing=config.dataset.preprocessing,
        include_paths=config.dataset.include_paths,
        window_size=config.dataset.window_size,
        stride=config.dataset.stride,
        normalization_stats=normalization_stats,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dim = dataset.target_dim
    model = create_model(config.model_name, output_dim=output_dim, **config.model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    print(f"Evaluating {config.model_name} on {device} with {len(dataset)} samples.")
    with torch.no_grad():
        progress = tqdm(data_loader, desc="Evaluation", leave=False)
        for emg_batch, pose_batch in progress:
            reset_model_state(model)
            prediction = model(emg_batch.to(device)).cpu().numpy()
            predictions.append(prediction)
            targets.append(pose_batch.numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metric_predictions = y_pred
    metric_targets = y_true
    if normalization_stats is not None and normalization_stats.has_target_stats():
        metric_predictions = invert_standardization(
            y_pred,
            normalization_stats.target_mean,
            normalization_stats.target_std,
        )
        metric_targets = invert_standardization(
            y_true,
            normalization_stats.target_mean,
            normalization_stats.target_std,
        )

    rmse = float(np.sqrt(mean_squared_error(metric_targets, metric_predictions)))
    mae = float(mean_absolute_error(metric_targets, metric_predictions))
    runtime_seconds = time.perf_counter() - start_time
    print(f"Evaluation complete in {runtime_seconds:.2f}s. rmse={rmse:.4f}, mae={mae:.4f}")
    return {
        "dataset_size": len(dataset),
        "rmse": rmse,
        "mae": mae,
        "model_name": config.model_name,
        "target_mode": config.dataset.preprocessing.target_mode,
        "target_representation": config.dataset.preprocessing.target_representation,
        "runtime_seconds": runtime_seconds,
    }
