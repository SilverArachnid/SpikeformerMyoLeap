"""Reusable model training loop."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from spikeformer_myo_leap.models import create_model

from .config import TrainingConfig
from .datasets import build_dataset_splits


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Train a model defined by ``config`` and return summary artifacts."""

    device = resolve_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    full_dataset, train_dataset, val_dataset = build_dataset_splits(
        dataset_root=config.dataset.dataset_root,
        preprocessing=config.dataset.preprocessing,
        include_paths=config.dataset.include_paths,
        window_size=config.dataset.window_size,
        stride=config.dataset.stride,
        train_fraction=config.split.train_fraction,
        seed=config.split.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dim = 63 if config.dataset.preprocessing.target_mode == "xyz" else 42
    model = create_model(config.model_name, output_dim=output_dim, **config.model_kwargs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    history = {"train_losses": [], "val_losses": []}
    best_checkpoint = os.path.join(config.output_dir, f"{config.model_name}_best.pt")
    last_checkpoint = os.path.join(config.output_dir, f"{config.model_name}_last.pt")

    for _epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0
        for emg_batch, pose_batch in train_loader:
            emg_batch = emg_batch.to(device)
            pose_batch = pose_batch.to(device)

            optimizer.zero_grad()
            prediction = model(emg_batch)
            loss = criterion(prediction, pose_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / max(1, len(train_loader))
        history["train_losses"].append(train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for emg_batch, pose_batch in val_loader:
                emg_batch = emg_batch.to(device)
                pose_batch = pose_batch.to(device)
                prediction = model(emg_batch)
                total_val_loss += criterion(prediction, pose_batch).item()

        val_loss = total_val_loss / max(1, len(val_loader))
        history["val_losses"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint)

    torch.save(model.state_dict(), last_checkpoint)

    summary = {
        "dataset_size": len(full_dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoints": {"best": best_checkpoint, "last": last_checkpoint},
        "model_name": config.model_name,
        "target_mode": config.dataset.preprocessing.target_mode,
        "config": asdict(config),
    }
    with open(os.path.join(config.output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
