"""Full-episode validation helpers for temporal and qualitative model assessment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn

from spikeformer_myo_leap.data import HAND_CONNECTIONS
from spikeformer_myo_leap.data.preprocessing import PreprocessedEpisode


@dataclass
class FullEpisodeValidationResult:
    """Metrics and saved output metadata for one held-out episode."""

    episode_dir: str
    valid_frame_count: int
    rmse: float
    mae: float
    visualization_path: str | None


def predict_full_episode(
    *,
    model: nn.Module,
    episode: PreprocessedEpisode,
    window_size: int,
    device: torch.device,
    reset_model_state: callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run rolling-window prediction over a whole preprocessed episode."""

    pose_dim = episode.pose.shape[1]
    predictions = np.full((episode.emg.shape[0], pose_dim), np.nan, dtype=np.float32)
    valid_mask = np.zeros((episode.emg.shape[0],), dtype=bool)

    model.eval()
    with torch.no_grad():
        for target_index in range(window_size, episode.emg.shape[0]):
            reset_model_state(model)
            window = episode.emg[target_index - window_size : target_index]
            emg_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            predictions[target_index] = model(emg_tensor).cpu().numpy()[0].astype(np.float32)
            valid_mask[target_index] = True

    return predictions, episode.pose.astype(np.float32), valid_mask


def _reshape_pose_for_plotting(frame: np.ndarray, target_mode: str) -> np.ndarray:
    """Reshape a flat pose vector into joint coordinates for visualization.

    Args:
        frame: Flat pose vector for one frame.
        target_mode: Pose representation mode. Currently only ``"xyz"`` is supported
            for 3D visualization.

    Raises:
        ValueError: If ``target_mode`` is not supported by the 3D plotting pipeline.
    """

    if target_mode != "xyz":
        raise ValueError(
            "Full-episode 3D visualization only supports target_mode='xyz'. "
            f"Received target_mode={target_mode!r}."
        )
    return frame.reshape(-1, 3)


def _axis_limits(targets: np.ndarray, predictions: np.ndarray, target_mode: str) -> tuple[float, float]:
    """Compute symmetric plot bounds for 3D skeleton visualization."""

    if target_mode != "xyz":
        raise ValueError(
            "Full-episode 3D visualization only supports target_mode='xyz'. "
            f"Received target_mode={target_mode!r}."
        )
    combined = np.concatenate([targets.reshape(-1, 3), predictions.reshape(-1, 3)], axis=0)
    bound = float(max(abs(np.nanmin(combined)), abs(np.nanmax(combined)), 1e-3))
    return -bound, bound


def save_episode_gif(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str,
    target_mode: str,
) -> None:
    """Save a side-by-side predicted-vs-ground-truth 3D skeleton GIF.

    Raises:
        ValueError: If ``target_mode`` is not supported by the visualization path.
    """

    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return

    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]
    lower, upper = _axis_limits(target_valid, pred_valid, target_mode)

    figure = plt.figure(figsize=(10, 5), facecolor="#020617")
    pred_ax = figure.add_subplot(1, 2, 1, projection="3d")
    target_ax = figure.add_subplot(1, 2, 2, projection="3d")

    def style_axis(ax: Any) -> None:
        """Apply a consistent dark theme to a 3D matplotlib axis."""

        ax.set_facecolor("#0f172a")
        ax.xaxis.pane.set_facecolor((15 / 255, 23 / 255, 42 / 255, 1.0))
        ax.yaxis.pane.set_facecolor((15 / 255, 23 / 255, 42 / 255, 1.0))
        ax.zaxis.pane.set_facecolor((15 / 255, 23 / 255, 42 / 255, 1.0))
        ax.xaxis.pane.set_edgecolor("#334155")
        ax.yaxis.pane.set_edgecolor("#334155")
        ax.zaxis.pane.set_edgecolor("#334155")
        ax.tick_params(colors="#cbd5e1")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.zaxis.label.set_color("#cbd5e1")

    def draw_hand(
        ax: Any,
        pose_frame: np.ndarray,
        title: str,
        line_color: str,
        joint_color: str,
    ) -> None:
        ax.clear()
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_zlim(lower, upper)
        ax.set_title(title, color="#e2e8f0")
        style_axis(ax)
        joints = _reshape_pose_for_plotting(pose_frame, target_mode)
        for start_idx, end_idx in HAND_CONNECTIONS:
            segment = joints[[start_idx, end_idx]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=line_color, linewidth=2)
        ax.scatter(
            joints[:, 0],
            joints[:, 1],
            joints[:, 2],
            color=joint_color,
            edgecolors="#020617",
            linewidths=0.5,
            s=24,
        )

    def update(frame_index: int) -> list[Any]:
        valid_index = valid_indices[frame_index]
        draw_hand(pred_ax, predictions[valid_index], "Predicted", "#38bdf8", "#f59e0b")
        draw_hand(target_ax, targets[valid_index], "Ground Truth", "#22c55e", "#f8fafc")
        figure.suptitle(f"Frame {valid_index}", color="#e2e8f0")
        return []

    ani = animation.FuncAnimation(figure, update, frames=len(valid_indices), interval=80, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=12))
    plt.close(figure)


def run_full_episode_validation(
    *,
    model: nn.Module,
    episodes: list[PreprocessedEpisode],
    device: torch.device,
    window_size: int,
    output_dir: str,
    epoch_index: int,
    config: dict[str, object],
    reset_model_state: callable,
) -> list[FullEpisodeValidationResult]:
    """Evaluate a fixed subset of held-out episodes and optionally save GIFs."""

    if not config.get("enabled", False):
        return []

    every_n_epochs = int(config.get("every_n_epochs", 1))
    if every_n_epochs <= 0 or (epoch_index + 1) % every_n_epochs != 0:
        return []

    episode_limit = min(int(config.get("num_episodes", 1)), len(episodes))
    selected_episodes = episodes[:episode_limit]
    save_visualizations = bool(config.get("save_visualizations", True))
    epoch_dir = os.path.join(output_dir, "full_episode_eval", f"epoch_{epoch_index + 1:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    results: list[FullEpisodeValidationResult] = []
    for episode_number, episode in enumerate(selected_episodes, start=1):
        predictions, targets, valid_mask = predict_full_episode(
            model=model,
            episode=episode,
            window_size=window_size,
            device=device,
            reset_model_state=reset_model_state,
        )
        pred_valid = predictions[valid_mask]
        target_valid = targets[valid_mask]
        rmse = float(np.sqrt(mean_squared_error(target_valid, pred_valid)))
        mae = float(mean_absolute_error(target_valid, pred_valid))

        visualization_path = None
        if save_visualizations and episode.target_mode == "xyz":
            visualization_path = os.path.join(epoch_dir, f"episode_{episode_number:02d}.gif")
            save_episode_gif(
                predictions=predictions,
                targets=targets,
                valid_mask=valid_mask,
                output_path=visualization_path,
                target_mode=episode.target_mode,
            )

        results.append(
            FullEpisodeValidationResult(
                episode_dir=episode.episode_dir,
                valid_frame_count=int(valid_mask.sum()),
                rmse=rmse,
                mae=mae,
                visualization_path=visualization_path,
            )
        )
    return results
