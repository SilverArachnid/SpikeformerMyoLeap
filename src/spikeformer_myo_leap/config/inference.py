"""Configuration types for live inference runtime."""

from dataclasses import dataclass


@dataclass
class LiveInferenceConfig:
    """Runtime settings for checkpoint-backed live Myo inference.

    Attributes:
        checkpoint_path: Path to a packaged training checkpoint containing model
            weights, normalization stats, and training config.
        device: Torch device selection. ``"auto"`` prefers CUDA when available.
            Defaults to ``"auto"``.
        viewer: Runtime visualization backend. ``"local"`` uses the existing
            Matplotlib dashboard. ``"none"`` disables visualization. Defaults to
            ``"local"``.
        update_hz: Maximum live inference/update rate. EMG samples may arrive
            faster or slower than this; inference is throttled to this rate.
            Defaults to ``30.0``.
        emg_history_seconds: Maximum retained raw EMG history in seconds for
            online interpolation. Defaults to ``5.0``.
        prosthetic_model: Optional retargeting target. ``"none"`` keeps the
            runtime in biological-hand prediction mode only. Supported adapter
            names now include ``"ability_hand"``, ``"dexhandv2_right"``, and
            ``"dexhandv2_cobot_right"``. This first pass emits mapped joint
            commands but does not yet drive a simulator backend.
        simulator_backend: Optional simulator target. ``"none"`` disables
            simulation, while ``"mujoco"`` drives a MuJoCo hand model from the
            retargeted prosthetic joint commands.
        simulator_model_path: Path to the MuJoCo XML/MJCF/URDF hand model when
            ``simulator_backend="mujoco"`` is selected. For the DexHand targets,
            this may be left empty when the sibling legacy ``SpikeFormerMyo``
            repo is present locally.
        show_emg: Whether to include the live Myo EMG panel in the local viewer.
            Defaults to ``True``.
    """

    checkpoint_path: str = ""
    device: str = "auto"
    viewer: str = "local"
    update_hz: float = 30.0
    emg_history_seconds: float = 5.0
    prosthetic_model: str = "none"
    simulator_backend: str = "none"
    simulator_model_path: str = ""
    show_emg: bool = True
