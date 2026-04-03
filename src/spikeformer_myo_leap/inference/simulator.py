"""Optional simulator backends for live prosthetic retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time

import numpy as np

from spikeformer_myo_leap.inference.prosthetics import ProstheticCommandSet


@dataclass(frozen=True)
class SimulatorFrameStats:
    """Lightweight runtime stats for a simulator backend."""

    frame_count: int = 0
    fps: float = 0.0


class SimulatorBackend:
    """Abstract backend interface for prosthetic simulation."""

    def apply(self, command_set: ProstheticCommandSet) -> SimulatorFrameStats:
        raise NotImplementedError

    def latest_frame(self) -> np.ndarray | None:
        """Return the latest RGB frame if the backend renders one."""

        return None

    def close(self) -> None:
        """Release any backend resources."""


class NoOpSimulatorBackend(SimulatorBackend):
    """Backend that intentionally does nothing."""

    def apply(self, command_set: ProstheticCommandSet) -> SimulatorFrameStats:
        del command_set
        return SimulatorFrameStats()


class MujocoSimulatorBackend(SimulatorBackend):
    """Lazy MuJoCo backend for live hand retargeting."""

    def __init__(self, model_path: str) -> None:
        os.environ.setdefault("MUJOCO_GL", "glfw")
        try:
            import mujoco
            import mujoco.viewer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MuJoCo backend requested but the 'mujoco' package is not installed."
            ) from exc

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        offwidth = max(1, int(self._model.vis.global_.offwidth))
        offheight = max(1, int(self._model.vis.global_.offheight))
        self._renderer = mujoco.Renderer(self._model, height=offheight, width=offwidth)
        self._camera = mujoco.MjvCamera()
        self._configure_camera()
        self._mujoco.mj_forward(self._model, self._data)
        self._last_frame: np.ndarray | None = None
        self._actuator_names = {
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id): actuator_id
            for actuator_id in range(self._model.nu)
        }
        self._joint_names = {
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): joint_id
            for joint_id in range(self._model.njnt)
        }
        self._frame_count = 0
        self._ema_fps = 0.0

    def _configure_camera(self) -> None:
        """Frame the hand model explicitly instead of relying on viewer defaults."""

        extent = max(float(self._model.stat.extent), 1e-3)
        center = self._model.stat.center
        self._camera.lookat[:] = center
        self._camera.distance = extent * 2.2
        self._camera.azimuth = 145.0
        self._camera.elevation = -25.0

    def apply(self, command_set: ProstheticCommandSet) -> SimulatorFrameStats:
        start_s = time.perf_counter()
        for joint_name, value in command_set.joint_targets.items():
            actuator_id = self._actuator_names.get(joint_name)
            if actuator_id is not None:
                self._data.ctrl[actuator_id] = value
                continue

            joint_id = self._joint_names.get(joint_name)
            if joint_id is not None:
                qpos_address = self._model.jnt_qposadr[joint_id]
                self._data.qpos[qpos_address] = value

        self._mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera)
        self._last_frame = self._renderer.render().copy()

        elapsed_s = max(time.perf_counter() - start_s, 1e-6)
        instantaneous_fps = 1.0 / elapsed_s
        self._ema_fps = instantaneous_fps if self._frame_count == 0 else (self._ema_fps * 0.9 + instantaneous_fps * 0.1)
        self._frame_count += 1
        return SimulatorFrameStats(frame_count=self._frame_count, fps=self._ema_fps)

    def latest_frame(self) -> np.ndarray | None:
        return None if self._last_frame is None else self._last_frame.copy()

    def close(self) -> None:
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_legacy_repo_root() -> Path:
    return _repo_root().parent / "SpikeFormerMyo"


def _vendored_assets_root() -> Path:
    return _repo_root() / "assets" / "prosthetics" / "dexhandv2"


def _prepare_dexhand_urdf(prosthetic_model: str) -> Path:
    vendored_root = _vendored_assets_root()
    legacy_root = _default_legacy_repo_root()

    if prosthetic_model == "dexhandv2_right":
        mesh_subdir = Path("meshes/right")
        package_prefix = "package://dexhandv2_description/meshes/right/"
    elif prosthetic_model == "dexhandv2_cobot_right":
        mesh_subdir = Path("meshes/cobot_right")
        package_prefix = "package://dexhandv2_description/meshes/cobot_right/"
    else:
        raise ValueError(f"{prosthetic_model!r} is not a bundled DexHand target.")

    vendored_urdf = vendored_root / "urdf" / f"{prosthetic_model}.urdf"
    vendored_mesh_dir = vendored_root / mesh_subdir
    if vendored_urdf.exists() and vendored_mesh_dir.exists():
        urdf_path = vendored_urdf
        mesh_dir = vendored_mesh_dir
    else:
        urdf_path = legacy_root / "urdf" / f"{prosthetic_model}.urdf"
        mesh_dir = legacy_root / mesh_subdir

    if not urdf_path.exists() or not mesh_dir.exists():
        raise FileNotFoundError(
            f"Could not resolve default DexHand assets for {prosthetic_model!r} "
            f"under {vendored_root} or {legacy_root}."
        )

    urdf_text = urdf_path.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace(package_prefix, "")
    urdf_text = re.sub(r"\s*<inertial>.*?</inertial>", "", urdf_text, flags=re.DOTALL)
    compiler_tag = (
        f'\n  <mujoco>\n'
        f'    <compiler meshdir="{mesh_dir.as_posix()}" balanceinertia="true" '
        f'boundmass="1e-4" boundinertia="1e-8"/>\n'
        f'  </mujoco>'
    )
    urdf_text = urdf_text.replace(
        f'<robot name="{prosthetic_model}">',
        f'<robot name="{prosthetic_model}">{compiler_tag}',
        1,
    )

    cache_dir = _repo_root() / ".cache" / "mujoco_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    patched_path = cache_dir / f"{prosthetic_model}.urdf"
    patched_path.write_text(urdf_text, encoding="utf-8")
    return patched_path


def resolve_simulator_model_path(*, prosthetic_model: str, model_path: str) -> Path:
    """Resolve one simulator asset path from config plus local defaults."""

    if model_path:
        return Path(model_path).expanduser().resolve()
    if prosthetic_model in {"dexhandv2_right", "dexhandv2_cobot_right"}:
        return _prepare_dexhand_urdf(prosthetic_model)
    raise ValueError(
        "No simulator_model_path was provided, and no built-in simulator asset is available "
        f"for prosthetic_model={prosthetic_model!r}."
    )


def build_simulator_backend(*, backend_name: str, model_path: str, prosthetic_model: str) -> SimulatorBackend:
    """Construct one simulator backend by config."""

    if backend_name == "none":
        return NoOpSimulatorBackend()
    if backend_name == "mujoco":
        return MujocoSimulatorBackend(str(resolve_simulator_model_path(prosthetic_model=prosthetic_model, model_path=model_path)))
    raise ValueError(f"Unsupported simulator_backend={backend_name!r}.")
