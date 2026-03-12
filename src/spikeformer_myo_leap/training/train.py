"""Reusable model training loop."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikeformer_myo_leap.models import create_model

from .config import TrainingConfig
from .datasets import build_dataset_splits

try:
    from spikingjelly.clock_driven import functional as spiking_functional
except ImportError:  # pragma: no cover - optional for non-spiking models
    spiking_functional = None


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def reset_model_state(model: nn.Module) -> None:
    """Reset spiking network state when the model uses spiking neurons."""

    if spiking_functional is not None:
        spiking_functional.reset_net(model)


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Train a model defined by ``config`` and return summary artifacts."""

    run_start_time = time.perf_counter()
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
    print(
        f"Training {config.model_name} on {device} "
        f"with {len(train_dataset)} train samples and {len(val_dataset)} val samples."
    )

    for epoch_index in range(config.num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch_index + 1}/{config.num_epochs} [train]",
            leave=False,
        )
        for emg_batch, pose_batch in train_progress:
            reset_model_state(model)
            emg_batch = emg_batch.to(device)
            pose_batch = pose_batch.to(device)

            optimizer.zero_grad()
            prediction = model(emg_batch)
            loss = criterion(prediction, pose_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_train_loss / max(1, len(train_loader))
        history["train_losses"].append(train_loss)

        model.eval()
        total_val_loss = 0.0
        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch_index + 1}/{config.num_epochs} [val]",
            leave=False,
        )
        with torch.no_grad():
            for emg_batch, pose_batch in val_progress:
                reset_model_state(model)
                emg_batch = emg_batch.to(device)
                pose_batch = pose_batch.to(device)
                prediction = model(emg_batch)
                batch_val_loss = criterion(prediction, pose_batch).item()
                total_val_loss += batch_val_loss
                val_progress.set_postfix(loss=f"{batch_val_loss:.4f}")

        val_loss = total_val_loss / max(1, len(val_loader))
        history["val_losses"].append(val_loss)

        epoch_seconds = time.perf_counter() - epoch_start_time
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint)
            checkpoint_note = " | saved new best checkpoint"
        else:
            checkpoint_note = ""

        print(
            f"Epoch {epoch_index + 1}/{config.num_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"epoch_time={epoch_seconds:.2f}s{checkpoint_note}"
        )

    torch.save(model.state_dict(), last_checkpoint)
    total_runtime_seconds = time.perf_counter() - run_start_time

    summary = {
        "dataset_size": len(full_dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoints": {"best": best_checkpoint, "last": last_checkpoint},
        "model_name": config.model_name,
        "target_mode": config.dataset.preprocessing.target_mode,
        "runtime_seconds": total_runtime_seconds,
        "config": asdict(config),
    }
    with open(os.path.join(config.output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Training complete in {total_runtime_seconds:.2f}s.")
    return summary
