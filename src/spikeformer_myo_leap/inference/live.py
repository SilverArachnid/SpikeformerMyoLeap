"""Checkpoint-backed live Myo inference runtime."""

from __future__ import annotations

from collections import deque
import time
from typing import Any

import numpy as np
import pyomyo
import torch

from spikeformer_myo_leap.config import LiveInferenceConfig
from spikeformer_myo_leap.data import DatasetNormalizationStats, apply_standardization, invert_standardization
from spikeformer_myo_leap.inference.articulation import (
    format_articulation_status,
    joint_angles_to_canonical_articulation,
    points_to_canonical_articulation,
)
from spikeformer_myo_leap.inference.prosthetics import format_prosthetic_status, map_articulation_to_prosthetic
from spikeformer_myo_leap.inference.simulator import build_simulator_backend
from spikeformer_myo_leap.models import create_model
from spikeformer_myo_leap.training.train import reset_model_state, resolve_device
from spikeformer_myo_leap.visualization.local_dashboard import LocalDashboard


class OnlineEmgWindowBuilder:
    """Build fixed-rate live EMG windows from irregular live samples.

    Myo samples are buffered with timestamps and then interpolated onto the same
    resample rate used during training so the runtime window matches the offline
    preprocessing contract as closely as possible.
    """

    def __init__(self, *, resample_hz: float, window_size: int, history_seconds: float) -> None:
        self.resample_hz = resample_hz
        self.window_size = window_size
        self.history_seconds = history_seconds
        self._timestamps = deque()
        self._samples = deque()

    def append(self, timestamp_s: float, sample: np.ndarray) -> None:
        """Append one raw EMG sample and prune old history."""

        self._timestamps.append(float(timestamp_s))
        self._samples.append(np.asarray(sample, dtype=np.float32))
        cutoff = timestamp_s - self.history_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._samples.popleft()

    def has_window(self) -> bool:
        """Return whether enough history exists to build a full inference window."""

        if len(self._timestamps) < 2:
            return False
        required_span = (self.window_size - 1) / self.resample_hz
        return self._timestamps[-1] - self._timestamps[0] >= required_span

    def build_window(self) -> np.ndarray:
        """Return the most recent interpolated ``[window, 8]`` EMG slice."""

        if not self.has_window():
            raise ValueError("Not enough EMG history to build a live inference window.")

        timestamps = np.asarray(self._timestamps, dtype=np.float32)
        samples = np.vstack(self._samples).astype(np.float32)
        latest = timestamps[-1]
        required_span = (self.window_size - 1) / self.resample_hz
        target_times = np.linspace(latest - required_span, latest, self.window_size, dtype=np.float32)
        window = np.empty((self.window_size, samples.shape[1]), dtype=np.float32)
        for channel_index in range(samples.shape[1]):
            window[:, channel_index] = np.interp(target_times, timestamps, samples[:, channel_index])
        return window.astype(np.float32)


def _load_checkpoint_runtime(config: LiveInferenceConfig) -> tuple[dict[str, Any], torch.nn.Module, DatasetNormalizationStats | None, torch.device]:
    """Load a packaged training checkpoint and construct the runtime model."""

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set for live inference.")

    device = resolve_device(config.device)
    checkpoint_payload = torch.load(config.checkpoint_path, map_location=device)
    if not isinstance(checkpoint_payload, dict) or "model_state_dict" not in checkpoint_payload:
        raise ValueError(
            "Live inference requires a packaged checkpoint with model_state_dict, "
            "normalization_stats, and config metadata."
        )

    checkpoint_config = checkpoint_payload.get("config")
    if not checkpoint_config:
        raise ValueError("Checkpoint does not contain saved training config metadata.")

    normalization_stats = DatasetNormalizationStats.from_dict(checkpoint_payload.get("normalization_stats"))
    model_name = str(checkpoint_config["model_name"])
    model_kwargs = dict(checkpoint_config.get("model_kwargs", {}))
    preprocessing = checkpoint_config["dataset"]["preprocessing"]
    target_mode = str(preprocessing["target_mode"])
    target_representation = str(preprocessing.get("target_representation", "points"))
    if target_representation == "points":
        output_dim = 63 if target_mode == "xyz" else 42
    elif target_representation == "joint_angles":
        output_dim = 10
    else:
        raise ValueError(f"Unsupported target_representation={target_representation!r} in checkpoint.")

    model = create_model(model_name, output_dim=output_dim, **model_kwargs).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    return checkpoint_config, model, normalization_stats, device


def _prediction_to_hand_points(prediction: np.ndarray, target_mode: str) -> list[np.ndarray]:
    """Convert a point-model output vector into dashboard-ready hand points."""

    if target_mode == "xyz":
        return [prediction.reshape(21, 3).astype(np.float32)]
    if target_mode == "xy":
        xy = prediction.reshape(21, 2).astype(np.float32)
        xyz = np.concatenate([xy, np.zeros((21, 1), dtype=np.float32)], axis=1)
        return [xyz]
    raise ValueError(f"Unsupported target_mode={target_mode!r} for live hand visualization.")


def run_live_inference(config: LiveInferenceConfig) -> None:
    """Run live Myo inference using the preprocessing/model contract stored in a checkpoint."""

    checkpoint_config, model, normalization_stats, device = _load_checkpoint_runtime(config)
    preprocessing = checkpoint_config["dataset"]["preprocessing"]
    window_size = int(checkpoint_config["dataset"]["window_size"])
    resample_hz = float(preprocessing["resample_hz"])
    target_mode = str(preprocessing["target_mode"])
    target_representation = str(preprocessing.get("target_representation", "points"))
    model_name = str(checkpoint_config["model_name"])
    if config.prosthetic_model != "none" and target_mode != "xyz":
        raise ValueError("prosthetic retargeting currently requires xyz checkpoints.")
    if config.simulator_backend != "none" and config.prosthetic_model == "none":
        raise ValueError("simulator_backend requires a prosthetic_model to be selected.")

    if config.simulator_backend != "none":
        print(
            f"Launching {config.simulator_backend} simulator for prosthetic_model={config.prosthetic_model} "
            f"(model_path={config.simulator_model_path or 'auto'})"
        )
    simulator = build_simulator_backend(
        backend_name=config.simulator_backend,
        model_path=config.simulator_model_path,
        prosthetic_model=config.prosthetic_model,
    )
    if config.simulator_backend != "none":
        print(f"{config.simulator_backend} simulator launched.")

    dashboard = None
    if config.viewer == "local":
        dashboard = LocalDashboard(
            "SpikeformerMyoLeap  |  Live Inference",
            show_hand=(target_representation == "points"),
            show_emg=config.show_emg,
            show_angles=(target_representation == "joint_angles" or config.prosthetic_model != "none"),
            show_simulator=(config.simulator_backend != "none"),
        )
        dashboard.start()
        dashboard.update_status(
            mode="Live Inference",
            recording=False,
            pose_name=target_representation,
            subject_id=model_name,
            episode_label="Live",
            status_line="Waiting for Myo EMG history",
        )

    window_builder = OnlineEmgWindowBuilder(
        resample_hz=resample_hz,
        window_size=window_size,
        history_seconds=max(config.emg_history_seconds, (window_size / resample_hz) * 2.0),
    )
    last_inference_time = 0.0
    inference_count = 0
    ema_fps = 0.0
    latest_prediction: np.ndarray | None = None
    sample_count = 0

    print(f"Loading checkpoint: {config.checkpoint_path}")
    print(
        f"Running live inference for model={model_name}, target_mode={target_mode}, "
        f"target_representation={target_representation}, window_size={window_size}, resample_hz={resample_hz}"
    )

    myo = pyomyo.Myo(mode=pyomyo.emg_mode.PREPROCESSED)
    myo.connect()
    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    def myo_handler(emg: list[float], movement: Any) -> None:
        nonlocal sample_count, latest_prediction, last_inference_time, inference_count, ema_fps
        del movement
        now = time.perf_counter()
        sample = np.asarray(emg, dtype=np.float32)
        sample_count += 1
        window_builder.append(now, sample)
        if dashboard is not None and config.show_emg:
            dashboard.update_emg(sample)

        min_interval = 1.0 / max(config.update_hz, 1e-6)
        if not window_builder.has_window() or (now - last_inference_time) < min_interval:
            return

        last_inference_time = now
        infer_start = time.perf_counter()
        window = window_builder.build_window()
        if normalization_stats is not None and normalization_stats.has_emg_stats():
            window = apply_standardization(
                window,
                normalization_stats.emg_mean,
                normalization_stats.emg_std,
            )

        with torch.no_grad():
            reset_model_state(model)
            emg_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            prediction = model(emg_tensor).cpu().numpy()[0].astype(np.float32)

        if normalization_stats is not None and normalization_stats.has_target_stats():
            prediction = invert_standardization(
                prediction,
                normalization_stats.target_mean,
                normalization_stats.target_std,
            )

        infer_seconds = max(time.perf_counter() - infer_start, 1e-6)
        instantaneous_fps = 1.0 / infer_seconds
        ema_fps = instantaneous_fps if inference_count == 0 else (ema_fps * 0.9 + instantaneous_fps * 0.1)
        inference_count += 1
        latest_prediction = prediction

        if target_representation == "points":
            articulation = points_to_canonical_articulation(prediction, target_mode) if target_mode == "xyz" else None
            angle_display = articulation.as_array() if articulation is not None else None
        else:
            articulation = joint_angles_to_canonical_articulation(prediction)
            angle_display = prediction if config.prosthetic_model == "none" else articulation.as_array()

        prosthetic_commands = None if articulation is None else map_articulation_to_prosthetic(articulation, config.prosthetic_model)
        simulator_stats = None
        if prosthetic_commands is not None:
            simulator_stats = simulator.apply(prosthetic_commands)
            if dashboard is not None:
                simulator_frame = simulator.latest_frame()
                if simulator_frame is not None:
                    dashboard.update_simulator_frame(simulator_frame)

        if dashboard is not None:
            status_line = f"Live inference {ema_fps:.1f} FPS | resample={resample_hz:.0f}Hz window={window_size}"
            if target_representation == "points":
                dashboard.update_hand(_prediction_to_hand_points(prediction, target_mode))
                if angle_display is not None and config.prosthetic_model != "none":
                    dashboard.update_angles(angle_display)
                    status_line = f"{status_line} | {format_articulation_status(articulation)}"
            else:
                dashboard.update_angles(angle_display if angle_display is not None else prediction)
                status_line = f"{status_line} | {format_articulation_status(articulation)}"
            if prosthetic_commands is not None:
                status_line = f"{status_line} | {format_prosthetic_status(prosthetic_commands)}"
            if simulator_stats is not None and config.simulator_backend != "none":
                status_line = f"{status_line} | sim {simulator_stats.fps:.1f} FPS"
            dashboard.update_status(
                mode="Live Inference",
                recording=False,
                pose_name=f"{target_representation} -> {config.prosthetic_model}" if config.prosthetic_model != "none" else target_representation,
                subject_id=model_name,
                episode_label=f"{sample_count} EMG samples",
                sample_count_emg=sample_count,
                sample_count_pose=inference_count,
                status_line=status_line,
            )

    myo.add_emg_handler(myo_handler)
    print("Connected to Myo. Streaming live inference. Press Ctrl+C to stop.")

    try:
        while True:
            myo.run()
    except KeyboardInterrupt:
        pass
    finally:
        myo.disconnect()
        simulator.close()
        if dashboard is not None:
            dashboard.close()
        if latest_prediction is not None:
            print(f"Last prediction shape: {tuple(latest_prediction.shape)}")
        print(f"Processed {sample_count} EMG samples and {inference_count} inference steps.")
