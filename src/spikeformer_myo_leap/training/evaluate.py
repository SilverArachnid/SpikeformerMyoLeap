"""Reusable checkpoint evaluation logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from spikeformer_myo_leap.models import create_model

from .config import EvaluationConfig
from .datasets import build_windowed_dataset
from .train import resolve_device


def evaluate_model(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate a saved checkpoint against the current dataset config."""

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set before running evaluation.")

    device = resolve_device(config.device)
    dataset = build_windowed_dataset(
        dataset_root=config.dataset.dataset_root,
        preprocessing=config.dataset.preprocessing,
        include_paths=config.dataset.include_paths,
        window_size=config.dataset.window_size,
        stride=config.dataset.stride,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dim = 63 if config.dataset.preprocessing.target_mode == "xyz" else 42
    model = create_model(config.model_name, output_dim=output_dim, **config.model_kwargs).to(device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for emg_batch, pose_batch in data_loader:
            prediction = model(emg_batch.to(device)).cpu().numpy()
            predictions.append(prediction)
            targets.append(pose_batch.numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "dataset_size": len(dataset),
        "rmse": rmse,
        "mae": mae,
        "model_name": config.model_name,
        "target_mode": config.dataset.preprocessing.target_mode,
    }
