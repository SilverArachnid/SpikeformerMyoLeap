"""Worker-process client for collection control.

This keeps the Qt GUI process separate from the hardware/control process so a
sensor or controller stall does not freeze the window itself.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any

from spikeformer_myo_leap.collection.controller import CollectionController
from spikeformer_myo_leap.data.contracts import CollectionSettings


DEFAULT_STATUS_SNAPSHOT: dict[str, Any] = {
    "hardware_running": False,
    "myo_connected": False,
    "leap_connected": False,
    "myo_streaming": False,
    "leap_streaming": False,
    "myo_serial": "",
    "leap_serial": "",
    "recording": False,
    "finalizing_episode": False,
    "session_active": False,
    "mode": "Idle",
    "status_message": "Ready",
    "sample_count_emg": 0,
    "sample_count_pose": 0,
    "completed_episodes": 0,
    "episodes_per_session": 0,
    "last_saved_episode": "",
    "last_aborted_episode": "",
    "current_episode_label": "0 / 0",
    "last_error": "",
    "subject_id": "user_1",
    "session_name": "session_1",
    "pose_name": "test_pose",
    "episode_duration": 5.0,
    "save_dir": "datasets",
    "session_dir": "",
    "last_episode_path": "",
}

DEFAULT_PREVIEW_SNAPSHOT: dict[str, Any] = {
    "hand_points": [],
    "emg_window": [],
}


def _worker_main(
    command_queue: mp.Queue,
    response_queue: mp.Queue,
    status_queue: mp.Queue,
    preview_queue: mp.Queue,
) -> None:
    controller = CollectionController(visualize=False)

    def publish_status() -> None:
        try:
            status_queue.put_nowait(controller.get_status_snapshot())
        except Exception:
            pass

    def publish_preview() -> None:
        try:
            preview_queue.put_nowait(controller.get_preview_snapshot())
        except Exception:
            pass

    try:
        publish_status()
        publish_preview()
        last_status_publish = 0.0
        last_preview_publish = 0.0
        while True:
            now = time.perf_counter()
            if now - last_status_publish >= 0.2:
                publish_status()
                last_status_publish = now
            if now - last_preview_publish >= 0.1:
                publish_preview()
                last_preview_publish = now

            try:
                command = command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            command_id = command["id"]
            name = command["name"]
            payload = command.get("payload", {})

            try:
                if name == "shutdown":
                    controller.close()
                    response_queue.put({"id": command_id, "ok": True, "result": None})
                    publish_status()
                    publish_preview()
                    break
                if name == "set_settings":
                    controller.set_settings(CollectionSettings(**payload))
                    result = None
                elif name == "connect":
                    controller.connect()
                    result = None
                elif name == "disconnect":
                    controller.disconnect()
                    result = None
                elif name == "start_session":
                    controller.start_session(CollectionSettings(**payload))
                    result = None
                elif name == "stop_session":
                    controller.stop_session()
                    result = None
                elif name == "start_episode":
                    controller.start_episode()
                    result = None
                elif name == "stop_episode":
                    controller.stop_episode()
                    result = None
                elif name == "get_status_snapshot":
                    result = controller.get_status_snapshot()
                else:
                    raise RuntimeError(f"Unknown worker command: {name}")

                response_queue.put({"id": command_id, "ok": True, "result": result})
            except Exception as exc:
                response_queue.put({"id": command_id, "ok": False, "error": str(exc)})
            finally:
                publish_status()
                publish_preview()
    finally:
        try:
            controller.close()
        except Exception:
            pass


class CollectionWorkerClient:
    """Thin client facade used by the GUI.

    Commands are executed in a separate process so GUI responsiveness does not
    depend on the hardware/control stack staying healthy.
    """

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._response_lock = threading.Lock()
        self._cached_status = dict(DEFAULT_STATUS_SNAPSHOT)
        self._cached_preview = dict(DEFAULT_PREVIEW_SNAPSHOT)
        self._command_queue: mp.Queue | None = None
        self._response_queue: mp.Queue | None = None
        self._status_queue: mp.Queue | None = None
        self._preview_queue: mp.Queue | None = None
        self._process: mp.Process | None = None
        self._start_worker()

    def set_settings(self, settings: CollectionSettings) -> None:
        self._call("set_settings", asdict(settings))

    def connect(self) -> None:
        # pyomyo BLE discovery can take 10-15 s after a recent disconnect;
        # the default 5 s timeout is not enough and causes the "Timed out
        # waiting for worker command: connect" popup.
        self._call("connect", timeout=30.0)

    def disconnect(self) -> None:
        self._call("disconnect")

    def start_session(self, settings: CollectionSettings) -> None:
        self._call("start_session", asdict(settings))

    def stop_session(self) -> None:
        self._call("stop_session")

    def start_episode(self) -> None:
        self._call("start_episode")

    def stop_episode(self) -> None:
        self._call("stop_episode")

    def get_status_snapshot(self) -> dict[str, Any]:
        self._drain_status_queue()
        return dict(self._cached_status)

    def get_preview_snapshot(self) -> dict[str, Any]:
        self._drain_preview_queue()
        return dict(self._cached_preview)

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._call("shutdown", timeout=3.0)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None
        self._command_queue = None
        self._response_queue = None
        self._status_queue = None
        self._preview_queue = None

    def reset(self) -> None:
        self.close()
        self._cached_status = dict(DEFAULT_STATUS_SNAPSHOT)
        self._cached_preview = dict(DEFAULT_PREVIEW_SNAPSHOT)
        self._start_worker()

    def _start_worker(self) -> None:
        self._command_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._status_queue = self._ctx.Queue()
        self._preview_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._command_queue, self._response_queue, self._status_queue, self._preview_queue),
        )
        self._process.start()

    def _call(self, name: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
        if (
            self._process is None
            or self._command_queue is None
            or self._response_queue is None
            or self._status_queue is None
            or self._preview_queue is None
            or not self._process.is_alive()
        ):
            raise RuntimeError("Collection worker is not running.")
        command_id = str(uuid.uuid4())
        with self._response_lock:
            self._command_queue.put({"id": command_id, "name": name, "payload": payload or {}})
            deadline = time.monotonic() + timeout
            while True:
                self._drain_status_queue()
                self._drain_preview_queue()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Timed out waiting for worker command: {name}")
                try:
                    response = self._response_queue.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if response["id"] != command_id:
                    continue
                if response["ok"]:
                    return response.get("result")
                raise RuntimeError(response["error"])

    def _drain_status_queue(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                snapshot = self._status_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._cached_status = snapshot

    def _drain_preview_queue(self) -> None:
        if self._preview_queue is None:
            return
        while True:
            try:
                snapshot = self._preview_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._cached_preview = snapshot
