"""Collection controller for synchronized Leap + Myo recording."""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict

import leap
import numpy as np
import pyomyo

from spikeformer_myo_leap.data.contracts import CollectionSettings
from spikeformer_myo_leap.data.io import pose_dir, save_episode, session_dir
from spikeformer_myo_leap.visualization.local_dashboard import LocalDashboard

MYO_STALE_SECONDS = 1.0
LEAP_STALE_SECONDS = 1.0
MIN_RECORD_SECONDS = 0.25
MIN_EFFECTIVE_HZ = 5.0


def extract_hand_points(hand) -> np.ndarray:
    """Convert a Leap hand object into the canonical 21-joint XYZ layout."""

    wrist = hand.arm.next_joint
    points = [[wrist.x, wrist.y, wrist.z]]
    for digit in hand.digits:
        points.extend(
            [
                [digit.metacarpal.next_joint.x, digit.metacarpal.next_joint.y, digit.metacarpal.next_joint.z],
                [digit.proximal.next_joint.x, digit.proximal.next_joint.y, digit.proximal.next_joint.z],
                [digit.intermediate.next_joint.x, digit.intermediate.next_joint.y, digit.intermediate.next_joint.z],
                [digit.distal.next_joint.x, digit.distal.next_joint.y, digit.distal.next_joint.z],
            ]
        )
    return np.asarray(points, dtype=np.float32)


class LeapCollectorListener(leap.Listener):
    """Bridge Leap connection callbacks into the collection controller."""

    def __init__(self, controller: "CollectionController") -> None:
        super().__init__()
        self.controller = controller

    def on_connection_event(self, event) -> None:
        del event
        self.controller._update_runtime(leap_connected=True, status_message="Leap connected")

    def on_device_event(self, event) -> None:
        try:
            with event.device.open():
                info = event.device.get_info()
                serial = info.serial
        except leap.LeapCannotOpenDeviceError:
            serial = "unknown"
        self.controller._update_runtime(
            leap_connected=True,
            leap_serial=serial,
            status_message=f"Leap device {serial} ready",
        )

    def on_tracking_event(self, event) -> None:
        if not event.hands:
            return
        self.controller._handle_leap_points(extract_hand_points(event.hands[0]))


class CollectionController:
    """Own hardware state, live monitoring, and episode recording lifecycle."""

    def __init__(self, settings: CollectionSettings | None = None, visualize: bool = True) -> None:
        self.settings = settings or CollectionSettings()
        self.visualize = visualize

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._record_stop_event = threading.Event()
        self._record_abort_event = threading.Event()

        self._myo_thread: threading.Thread | None = None
        self._leap_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._record_thread: threading.Thread | None = None

        self._dashboard: LocalDashboard | None = None
        self._myo = None
        self._leap_connection = None
        self._leap_listener = None

        self._emg_data: list[tuple[float, ...]] = []
        self._pose_data: list[tuple[float, ...]] = []
        self._recording_start: float | None = None
        self._last_episode_path: str | None = None
        self._last_emg_sample_time: float | None = None
        self._last_pose_sample_time: float | None = None
        self._record_abort_reason = ""

        self.runtime = {
            "hardware_running": False,
            "myo_connected": False,
            "leap_connected": False,
            "myo_streaming": False,
            "leap_streaming": False,
            "myo_serial": "",
            "leap_serial": "",
            "recording": False,
            "session_active": False,
            "mode": "Idle",
            "status_message": "Ready",
            "sample_count_emg": 0,
            "sample_count_pose": 0,
            "completed_episodes": 0,
            "episodes_per_session": self.settings.episodes_per_session,
            "last_saved_episode": "",
            "last_aborted_episode": "",
            "current_episode_label": "0 / 0",
            "last_error": "",
        }
        self._refresh_session_counters()

    def set_settings(self, settings: CollectionSettings) -> None:
        """Replace the current collection settings."""

        with self._lock:
            self.settings = settings
            self.runtime["episodes_per_session"] = settings.episodes_per_session
            self._refresh_session_counters()
            self._push_dashboard_status()

    def connect(self) -> None:
        """Connect both sensors and start monitoring threads."""

        with self._lock:
            if self.runtime["hardware_running"]:
                return
            self._stop_event.clear()
            self._record_stop_event.clear()
            self._record_abort_event.clear()
            self._record_abort_reason = ""
            self._last_emg_sample_time = None
            self._last_pose_sample_time = None
            self.runtime["mode"] = "Connecting"
            self.runtime["status_message"] = "Initializing sensors"
            self.runtime["last_error"] = ""
            self.runtime["last_aborted_episode"] = ""
            self.runtime["myo_streaming"] = False
            self.runtime["leap_streaming"] = False
            if self.visualize and self._dashboard is None:
                self._dashboard = LocalDashboard(
                    "SpikeformerMyoLeap  |  Data Collection",
                    show_hand=True,
                    show_emg=True,
                )
                self._dashboard.start()
            self._push_dashboard_status()

        try:
            self._connect_myo()
            self._start_leap_thread()
            self._start_watchdog_thread()
        except Exception:
            self.disconnect()
            raise

        with self._lock:
            self.runtime["hardware_running"] = True
            self._update_ready_state_locked()
            self._push_dashboard_status()

    def start_session(self, settings: CollectionSettings) -> None:
        """Activate a session without starting an episode yet."""

        with self._lock:
            if not self.runtime["hardware_running"]:
                raise RuntimeError("Connect hardware before starting a session.")
            if self.runtime["recording"]:
                raise RuntimeError("Cannot change session while recording.")
        self.set_settings(settings)
        with self._lock:
            self.runtime["session_active"] = True
            self._refresh_session_counters()
            self._update_ready_state_locked()
            self._push_dashboard_status()

    def stop_session(self) -> None:
        """End the active session while keeping hardware connected."""

        with self._lock:
            if self.runtime["recording"]:
                raise RuntimeError("Stop the current recording before ending the session.")
            self.runtime["session_active"] = False
            self.runtime["status_message"] = "Session stopped"
            self._update_ready_state_locked()
            self._push_dashboard_status()

    def disconnect(self) -> None:
        """Stop threads, disconnect sensors, and close the dashboard."""

        self._stop_event.set()
        self._record_abort_reason = "Recording aborted during shutdown"
        self._record_abort_event.set()
        self._record_stop_event.set()

        if self._record_thread is not None and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.0)
        if self._myo_thread is not None and self._myo_thread.is_alive():
            self._myo_thread.join(timeout=1.0)
        if self._leap_thread is not None and self._leap_thread.is_alive():
            self._leap_thread.join(timeout=1.0)
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=1.0)

        self._record_thread = None
        self._myo_thread = None
        self._leap_thread = None
        self._watchdog_thread = None

        with self._lock:
            if self._myo is not None:
                try:
                    self._myo.disconnect()
                except Exception:
                    pass
                self._myo = None

            self._leap_connection = None
            self._leap_listener = None
            self._discard_current_episode_locked("")
            self.runtime["hardware_running"] = False
            self.runtime["myo_connected"] = False
            self.runtime["leap_connected"] = False
            self.runtime["myo_streaming"] = False
            self.runtime["leap_streaming"] = False
            self.runtime["session_active"] = False
            self.runtime["recording"] = False
            self.runtime["mode"] = "Disconnected"
            self.runtime["status_message"] = "Sensors disconnected"
            self._push_dashboard_status()

        if self._dashboard is not None:
            self._dashboard.close()
            self._dashboard = None

    def close(self) -> None:
        """Close the controller and release hardware resources."""

        self.disconnect()

    def start_episode(self) -> None:
        """Start recording the next synchronized episode."""

        with self._lock:
            if not self.runtime["hardware_running"]:
                raise RuntimeError("Connect sensors before recording.")
            if not self.runtime["session_active"]:
                raise RuntimeError("Start a session before recording.")
            if self.runtime["recording"]:
                raise RuntimeError("An episode is already being recorded.")
            if not self.runtime["myo_connected"] or not self.runtime["leap_connected"]:
                raise RuntimeError("Both Myo and Leap must be connected before recording.")
            if not self.runtime["myo_streaming"] or not self.runtime["leap_streaming"]:
                raise RuntimeError("Wait until both Myo and Leap data streams are healthy before recording.")
            if self.runtime["completed_episodes"] >= self.settings.episodes_per_session:
                raise RuntimeError("This session already reached the configured episode count.")

            self._emg_data = []
            self._pose_data = []
            self._recording_start = time.perf_counter()
            self._record_stop_event.clear()
            self._record_abort_event.clear()
            self._record_abort_reason = ""
            self.runtime["recording"] = True
            self.runtime["mode"] = "Recording"
            self.runtime["sample_count_emg"] = 0
            self.runtime["sample_count_pose"] = 0
            self.runtime["status_message"] = "Capturing synchronized Leap pose and Myo EMG"
            self._push_dashboard_status()

        self._record_thread = threading.Thread(target=self._record_episode_worker, daemon=True)
        self._record_thread.start()

    def stop_episode(self) -> None:
        """Stop the current episode early and save the partial capture."""

        with self._lock:
            if not self.runtime["recording"]:
                return
            self.runtime["status_message"] = "Stopping recording early and saving partial episode"
            self._push_dashboard_status()
            self._record_stop_event.set()

    def get_status_snapshot(self) -> dict[str, object]:
        """Return a UI-safe snapshot of runtime and settings."""

        with self._lock:
            snapshot = dict(self.runtime)
            snapshot.update(asdict(self.settings))
            snapshot["session_dir"] = self._session_dir()
            snapshot["last_episode_path"] = self._last_episode_path or ""
            return snapshot

    def _connect_myo(self) -> None:
        myo = pyomyo.Myo(mode=pyomyo.emg_mode.PREPROCESSED)
        myo.connect()
        myo.set_leds([0, 128, 0], [0, 128, 0])
        myo.vibrate(1)
        myo.add_emg_handler(self._handle_emg)

        with self._lock:
            self._myo = myo
            self.runtime["myo_connected"] = True
            self.runtime["status_message"] = "Myo connected"
            self._push_dashboard_status()

        self._myo_thread = threading.Thread(target=self._myo_loop, daemon=True)
        self._myo_thread.start()

    def _myo_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._myo is not None:
                    self._myo.run()
                else:
                    time.sleep(0.01)
            except Exception as exc:
                self._handle_stream_fault_locked(
                    sensor="myo",
                    message=f"Myo loop error: {exc}",
                )
                break

    def _start_leap_thread(self) -> None:
        self._leap_listener = LeapCollectorListener(self)
        self._leap_connection = leap.Connection()
        self._leap_connection.add_listener(self._leap_listener)
        self._leap_thread = threading.Thread(target=self._leap_loop, daemon=True)
        self._leap_thread.start()

    def _start_watchdog_thread(self) -> None:
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _leap_loop(self) -> None:
        try:
            with self._leap_connection.open():
                self._leap_connection.set_tracking_mode(leap.TrackingMode.Desktop)
                while not self._stop_event.is_set():
                    time.sleep(0.05)
        except Exception as exc:
            self._handle_stream_fault_locked(
                sensor="leap",
                message=f"Leap loop error: {exc}",
            )

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.1)
            with self._lock:
                if not self.runtime["hardware_running"]:
                    continue

                now = time.perf_counter()
                myo_streaming = self._last_emg_sample_time is not None and now - self._last_emg_sample_time <= MYO_STALE_SECONDS
                leap_streaming = self._last_pose_sample_time is not None and now - self._last_pose_sample_time <= LEAP_STALE_SECONDS
                stream_changed = (
                    myo_streaming != self.runtime["myo_streaming"]
                    or leap_streaming != self.runtime["leap_streaming"]
                )
                self.runtime["myo_streaming"] = myo_streaming
                self.runtime["leap_streaming"] = leap_streaming

                if self.runtime["recording"] and (not myo_streaming or not leap_streaming):
                    missing = []
                    if not myo_streaming:
                        missing.append("Myo")
                    if not leap_streaming:
                        missing.append("Leap")
                    self._abort_recording_locked(
                        f"Recording aborted because sensor data stopped: {', '.join(missing)}"
                    )
                    continue

                if stream_changed:
                    self._update_ready_state_locked()
                    self._push_dashboard_status()

    def _handle_emg(self, emg, movement) -> None:
        del movement
        with self._lock:
            self._last_emg_sample_time = time.perf_counter()
            self.runtime["myo_connected"] = True
            self.runtime["myo_streaming"] = True
            if self._dashboard is not None:
                self._dashboard.update_emg(emg)

            if self.runtime["recording"] and self._recording_start is not None:
                timestamp = (time.perf_counter() - self._recording_start) * 1000.0
                self._emg_data.append((timestamp, *emg))
                self.runtime["sample_count_emg"] = len(self._emg_data)
            else:
                self._update_ready_state_locked()

            self._push_dashboard_status()

    def _handle_leap_points(self, points: np.ndarray) -> None:
        joints_xyz = points.reshape(-1).tolist()
        with self._lock:
            self._last_pose_sample_time = time.perf_counter()
            self.runtime["leap_connected"] = True
            self.runtime["leap_streaming"] = True
            if self._dashboard is not None:
                self._dashboard.update_hand([points])

            if self.runtime["recording"] and self._recording_start is not None:
                timestamp = (time.perf_counter() - self._recording_start) * 1000.0
                self._pose_data.append((timestamp, *joints_xyz))
                self.runtime["sample_count_pose"] = len(self._pose_data)
            else:
                self._update_ready_state_locked()

            self._push_dashboard_status()

    def _record_episode_worker(self) -> None:
        duration = self.settings.episode_duration
        start = self._recording_start
        while (
            not self._stop_event.is_set()
            and not self._record_stop_event.is_set()
            and not self._record_abort_event.is_set()
            and start is not None
            and time.perf_counter() - start < duration
        ):
            time.sleep(0.01)

        with self._lock:
            abort_reason = self._record_abort_reason
            shutting_down = self._stop_event.is_set()
            self.runtime["recording"] = False
            if abort_reason or shutting_down:
                self.runtime["mode"] = "Recovering" if not shutting_down else "Disconnected"
                self.runtime["status_message"] = abort_reason or "Recording aborted during shutdown"
            else:
                self.runtime["mode"] = "Saving"
                self.runtime["status_message"] = "Writing episode files to disk"
            self._push_dashboard_status()

        if abort_reason or shutting_down:
            with self._lock:
                aborted_episode = self._next_episode_name_locked()
                self._discard_current_episode_locked(abort_reason or "Recording aborted during shutdown")
                self.runtime["last_aborted_episode"] = aborted_episode
                self._update_ready_state_locked()
                self._push_dashboard_status()
            return

        try:
            episode_path = self._save_current_episode()
        except Exception as exc:
            self._set_error(f"Save error: {exc}")
            return

        with self._lock:
            self._last_episode_path = episode_path
            self._refresh_session_counters()
            self.runtime["last_saved_episode"] = os.path.basename(episode_path)
            self.runtime["status_message"] = f"Saved {os.path.basename(episode_path)}"
            if self._myo is not None:
                try:
                    self._myo.vibrate(1)
                except Exception:
                    pass
            self._update_ready_state_locked()
            self.runtime["status_message"] = f"Saved {os.path.basename(episode_path)}"
            self._push_dashboard_status()

    def _save_current_episode(self) -> str:
        episode_number = self.runtime["completed_episodes"] + 1
        recorded_duration_seconds = (
            0.0 if self._recording_start is None else max(0.0, time.perf_counter() - self._recording_start)
        )
        self._validate_episode_capture(recorded_duration_seconds)
        episode_folder = save_episode(
            settings=self.settings,
            episode_number=episode_number,
            emg_data=self._emg_data,
            pose_data=self._pose_data,
            episode_id=str(uuid.uuid4()),
            recorded_duration_seconds=recorded_duration_seconds,
        )
        self._discard_current_episode_locked("")
        return episode_folder

    def _validate_episode_capture(self, recorded_duration_seconds: float) -> None:
        if recorded_duration_seconds < MIN_RECORD_SECONDS:
            raise ValueError("Recorded episode is too short to save.")
        if len(self._emg_data) == 0:
            raise ValueError("No Myo EMG samples were captured for this episode.")
        if len(self._pose_data) == 0:
            raise ValueError("No Leap pose samples were captured for this episode.")
        effective_emg_hz = len(self._emg_data) / max(recorded_duration_seconds, 1e-6)
        effective_pose_hz = len(self._pose_data) / max(recorded_duration_seconds, 1e-6)
        if effective_emg_hz < MIN_EFFECTIVE_HZ:
            raise ValueError(f"Myo sample rate is too low to save ({effective_emg_hz:.2f} Hz).")
        if effective_pose_hz < MIN_EFFECTIVE_HZ:
            raise ValueError(f"Leap sample rate is too low to save ({effective_pose_hz:.2f} Hz).")

    def _discard_current_episode_locked(self, reason: str) -> None:
        self._emg_data = []
        self._pose_data = []
        self._recording_start = None
        self._record_stop_event.clear()
        self._record_abort_event.clear()
        self._record_abort_reason = ""
        self.runtime["sample_count_emg"] = 0
        self.runtime["sample_count_pose"] = 0
        if reason:
            self.runtime["last_error"] = reason

    def _abort_recording_locked(self, reason: str) -> None:
        if not self.runtime["recording"]:
            return
        self._record_abort_reason = reason
        self._record_abort_event.set()
        self.runtime["mode"] = "Recovering"
        self.runtime["status_message"] = reason
        self.runtime["last_error"] = reason
        self._push_dashboard_status()

    def _handle_stream_fault_locked(self, sensor: str, message: str) -> None:
        with self._lock:
            if sensor == "myo":
                self.runtime["myo_connected"] = False
                self.runtime["myo_streaming"] = False
            elif sensor == "leap":
                self.runtime["leap_connected"] = False
                self.runtime["leap_streaming"] = False
            self.runtime["last_error"] = message
            if self.runtime["recording"]:
                self._abort_recording_locked(message)
            else:
                self.runtime["status_message"] = message
                self._update_ready_state_locked()
                self._push_dashboard_status()

    def _push_dashboard_status(self) -> None:
        if self._dashboard is None:
            return
        self._dashboard.update_status(
            mode=self.runtime["mode"],
            recording=self.runtime["recording"],
            pose_name=self.settings.pose_name,
            subject_id=self.settings.subject_id,
            episode_label=self.runtime["current_episode_label"],
            sample_count_pose=self.runtime["sample_count_pose"],
            sample_count_emg=self.runtime["sample_count_emg"],
            status_line=self.runtime["status_message"],
        )

    def _refresh_session_counters(self) -> None:
        root = self._pose_dir()
        completed = 0
        if os.path.isdir(root):
            completed = len(
                [
                    name
                    for name in os.listdir(root)
                    if name.startswith("ep_") and os.path.isdir(os.path.join(root, name))
                ]
            )
        self.runtime["completed_episodes"] = completed
        self.runtime["current_episode_label"] = f"{completed} / {self.settings.episodes_per_session}"

    def _session_dir(self) -> str:
        return session_dir(self.settings.save_dir, self.settings.subject_id, self.settings.session_name)

    def _pose_dir(self) -> str:
        return pose_dir(self.settings.save_dir, self.settings.subject_id, self.settings.session_name, self.settings.pose_name)

    def _next_episode_name_locked(self) -> str:
        return f"ep_{self.runtime['completed_episodes'] + 1:04d}"

    def _update_ready_state_locked(self) -> None:
        if self.runtime["recording"]:
            return
        if not self.runtime["hardware_running"]:
            self.runtime["mode"] = "Disconnected"
            self.runtime["status_message"] = "Sensors disconnected"
            return
        if not self.runtime["myo_connected"] or not self.runtime["leap_connected"]:
            self.runtime["mode"] = "Waiting For Sensors"
            self.runtime["status_message"] = "Waiting for both sensors to connect"
            return
        if not self.runtime["myo_streaming"] or not self.runtime["leap_streaming"]:
            waiting = []
            if not self.runtime["myo_streaming"]:
                waiting.append("Myo")
            if not self.runtime["leap_streaming"]:
                waiting.append("Leap")
            self.runtime["mode"] = "Waiting For Sensors"
            self.runtime["status_message"] = f"Waiting for sensor data: {', '.join(waiting)}"
            return
        if self.runtime["session_active"]:
            self.runtime["mode"] = "Session Ready"
            self.runtime["status_message"] = "Sensors healthy. Ready to record the next episode."
        else:
            self.runtime["mode"] = "Monitoring"
            self.runtime["status_message"] = "Sensors connected and streaming"

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.runtime["last_error"] = message
            self.runtime["status_message"] = message
            self.runtime["recording"] = False
            self.runtime["session_active"] = self.runtime["session_active"] and self.runtime["hardware_running"]
            self._update_ready_state_locked()
            self.runtime["last_error"] = message
            self.runtime["status_message"] = message
            self._push_dashboard_status()

    def _update_runtime(self, **updates) -> None:
        with self._lock:
            self.runtime.update(updates)
            self._push_dashboard_status()
