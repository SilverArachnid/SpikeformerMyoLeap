import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import leap
import numpy as np
import pandas as pd
import pyomyo

from local_visualizer import LocalDashboard


LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def extract_hand_points(hand):
    wrist = hand.arm.next_joint
    points = [[wrist.x, wrist.y, wrist.z]]
    for digit in hand.digits:
        points.extend([
            [digit.metacarpal.next_joint.x, digit.metacarpal.next_joint.y, digit.metacarpal.next_joint.z],
            [digit.proximal.next_joint.x, digit.proximal.next_joint.y, digit.proximal.next_joint.z],
            [digit.intermediate.next_joint.x, digit.intermediate.next_joint.y, digit.intermediate.next_joint.z],
            [digit.distal.next_joint.x, digit.distal.next_joint.y, digit.distal.next_joint.z],
        ])
    return np.asarray(points, dtype=np.float32)


@dataclass
class CollectionSettings:
    subject_id: str = "user_1"
    session_name: str = "session_1"
    pose_name: str = "test_pose"
    episode_duration: float = 5.0
    episodes_per_session: int = 20
    save_dir: str = "datasets"


class LeapCollectorListener(leap.Listener):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def on_connection_event(self, event):
        self.controller._update_runtime(leap_connected=True, status_message="Leap connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
                serial = info.serial
        except leap.LeapCannotOpenDeviceError:
            serial = "unknown"
        self.controller._update_runtime(leap_connected=True, leap_serial=serial, status_message=f"Leap device {serial} ready")

    def on_tracking_event(self, event):
        if not event.hands:
            return
        points = extract_hand_points(event.hands[0])
        self.controller._handle_leap_points(points)


class CollectionController:
    def __init__(self, settings=None, visualize=True):
        self.settings = settings or CollectionSettings()
        self.visualize = visualize

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._myo_thread = None
        self._leap_thread = None
        self._record_thread = None
        self._record_stop_event = threading.Event()
        self._dashboard = None
        self._myo = None
        self._leap_connection = None
        self._leap_listener = None

        self._emg_data = []
        self._pose_data = []
        self._recording_start = None
        self._last_episode_path = None

        self.runtime = {
            "hardware_running": False,
            "myo_connected": False,
            "leap_connected": False,
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
            "current_episode_label": "0 / 0",
            "last_error": "",
        }
        self._refresh_session_counters()

    def set_settings(self, settings):
        with self._lock:
            self.settings = settings
            self.runtime["episodes_per_session"] = settings.episodes_per_session
            self._refresh_session_counters()
            self._push_dashboard_status()

    def connect(self):
        with self._lock:
            if self.runtime["hardware_running"]:
                return
            self._stop_event.clear()
            self.runtime["mode"] = "Connecting"
            self.runtime["status_message"] = "Initializing sensors"
            self.runtime["last_error"] = ""
            if self.visualize and self._dashboard is None:
                self._dashboard = LocalDashboard("SpikeformerMyoLeap  |  Data Collection", show_hand=True, show_emg=True)
                self._dashboard.start()
            self._push_dashboard_status()

        try:
            self._connect_myo()
            self._start_leap_thread()
        except Exception:
            self.disconnect()
            raise

        with self._lock:
            self.runtime["hardware_running"] = True
            self.runtime["mode"] = "Monitoring"
            self.runtime["status_message"] = "Sensors connected and ready"
            self._push_dashboard_status()

    def start_session(self, settings):
        with self._lock:
            if not self.runtime["hardware_running"]:
                raise RuntimeError("Connect hardware before starting a session.")
            if self.runtime["recording"]:
                raise RuntimeError("Cannot change session while recording.")
        self.set_settings(settings)
        with self._lock:
            self.runtime["session_active"] = True
            self.runtime["mode"] = "Session Ready"
            self.runtime["status_message"] = "Session started. Ready to record the next episode."
            self._refresh_session_counters()
            self._push_dashboard_status()

    def stop_session(self):
        with self._lock:
            if self.runtime["recording"]:
                raise RuntimeError("Stop the current recording before ending the session.")
            self.runtime["session_active"] = False
            self.runtime["mode"] = "Monitoring" if self.runtime["hardware_running"] else "Disconnected"
            self.runtime["status_message"] = "Session stopped"
            self._push_dashboard_status()

    def disconnect(self):
        self._stop_event.set()
        self._record_stop_event.set()

        if self._record_thread is not None and self._record_thread.is_alive():
            self._record_thread.join(timeout=0.5)

        if self._myo_thread is not None and self._myo_thread.is_alive():
            self._myo_thread.join(timeout=1.0)
        self._myo_thread = None

        if self._leap_thread is not None and self._leap_thread.is_alive():
            self._leap_thread.join(timeout=1.0)
        self._leap_thread = None

        with self._lock:
            if self._myo is not None:
                try:
                    self._myo.disconnect()
                except Exception:
                    pass
                self._myo = None

            self._leap_connection = None
            self._leap_listener = None
            self.runtime["hardware_running"] = False
            self.runtime["myo_connected"] = False
            self.runtime["leap_connected"] = False
            self.runtime["recording"] = False
            self.runtime["session_active"] = False
            self.runtime["mode"] = "Disconnected"
            self.runtime["status_message"] = "Sensors disconnected"
            self._push_dashboard_status()

        if self._dashboard is not None:
            self._dashboard.close()
            self._dashboard = None

    def close(self):
        self.disconnect()

    def start_episode(self):
        with self._lock:
            if not self.runtime["hardware_running"]:
                raise RuntimeError("Connect sensors before recording.")
            if not self.runtime["session_active"]:
                raise RuntimeError("Start a session before recording.")
            if self.runtime["recording"]:
                raise RuntimeError("An episode is already being recorded.")
            if not self.runtime["myo_connected"] or not self.runtime["leap_connected"]:
                raise RuntimeError("Both Myo and Leap must be connected before recording.")
            if self.runtime["completed_episodes"] >= self.settings.episodes_per_session:
                raise RuntimeError("This session already reached the configured episode count.")

            self._emg_data = []
            self._pose_data = []
            self._recording_start = time.perf_counter()
            self._record_stop_event.clear()
            self.runtime["recording"] = True
            self.runtime["mode"] = "Recording"
            self.runtime["sample_count_emg"] = 0
            self.runtime["sample_count_pose"] = 0
            self.runtime["status_message"] = "Capturing synchronized Leap pose and Myo EMG"
            self._push_dashboard_status()

        self._record_thread = threading.Thread(target=self._record_episode_worker, daemon=True)
        self._record_thread.start()

    def stop_episode(self):
        with self._lock:
            if not self.runtime["recording"]:
                return
            self.runtime["status_message"] = "Stopping recording early and saving partial episode"
            self._push_dashboard_status()
            self._record_stop_event.set()

    def get_status_snapshot(self):
        with self._lock:
            snapshot = dict(self.runtime)
            snapshot.update(asdict(self.settings))
            snapshot["session_dir"] = self._session_dir()
            snapshot["last_episode_path"] = self._last_episode_path or ""
            return snapshot

    def _connect_myo(self):
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

    def _myo_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._myo is not None:
                    self._myo.run()
                else:
                    time.sleep(0.01)
            except Exception as exc:
                self._set_error(f"Myo loop error: {exc}")
                break

    def _start_leap_thread(self):
        self._leap_listener = LeapCollectorListener(self)
        self._leap_connection = leap.Connection()
        self._leap_connection.add_listener(self._leap_listener)
        self._leap_thread = threading.Thread(target=self._leap_loop, daemon=True)
        self._leap_thread.start()

    def _leap_loop(self):
        try:
            with self._leap_connection.open():
                self._leap_connection.set_tracking_mode(leap.TrackingMode.Desktop)
                while not self._stop_event.is_set():
                    time.sleep(0.05)
        except Exception as exc:
            self._set_error(f"Leap loop error: {exc}")

    def _handle_emg(self, emg, movement):
        with self._lock:
            if self._dashboard is not None:
                self._dashboard.update_emg(emg)

            if self.runtime["recording"] and self._recording_start is not None:
                timestamp = (time.perf_counter() - self._recording_start) * 1000.0
                self._emg_data.append((timestamp, *emg))
                self.runtime["sample_count_emg"] = len(self._emg_data)

            self._push_dashboard_status()

    def _handle_leap_points(self, points):
        joints_xyz = points.reshape(-1).tolist()
        with self._lock:
            if self._dashboard is not None:
                self._dashboard.update_hand([points])

            if self.runtime["recording"] and self._recording_start is not None:
                timestamp = (time.perf_counter() - self._recording_start) * 1000.0
                self._pose_data.append((timestamp, *joints_xyz))
                self.runtime["sample_count_pose"] = len(self._pose_data)

            self._push_dashboard_status()

    def _record_episode_worker(self):
        duration = self.settings.episode_duration
        start = self._recording_start
        while (
            not self._stop_event.is_set()
            and not self._record_stop_event.is_set()
            and start is not None
            and time.perf_counter() - start < duration
        ):
            time.sleep(0.01)

        with self._lock:
            self.runtime["recording"] = False
            self.runtime["mode"] = "Saving"
            self.runtime["status_message"] = "Writing episode files to disk"
            self._push_dashboard_status()

        try:
            episode_path = self._save_current_episode()
            with self._lock:
                self._last_episode_path = episode_path
                self._refresh_session_counters()
                self.runtime["mode"] = "Monitoring"
                self.runtime["status_message"] = f"Saved {os.path.basename(episode_path)}"
                self.runtime["last_saved_episode"] = os.path.basename(episode_path)
                self.runtime["session_active"] = True
                if self._myo is not None:
                    try:
                        self._myo.vibrate(1)
                    except Exception:
                        pass
                self._push_dashboard_status()
        except Exception as exc:
            self._set_error(f"Save error: {exc}")

    def _save_current_episode(self):
        episode_number = self.runtime["completed_episodes"] + 1
        episode_folder = self._episode_dir(episode_number)
        os.makedirs(episode_folder, exist_ok=False)

        if self._emg_data:
            ts_emg, *channels = zip(*self._emg_data)
            emg_arr = np.array(channels).T
            emg_df = pd.DataFrame(emg_arr, columns=[f"Channel_{i+1}" for i in range(8)])
            emg_df.insert(0, "Timestamp_ms", ts_emg)
            emg_df.to_csv(os.path.join(episode_folder, "emg.csv"), index=False)

        if self._pose_data:
            ts_pose, *pose_pts = zip(*self._pose_data)
            pose_arr = np.array(pose_pts).T
            columns = []
            for name in LANDMARK_NAMES:
                columns.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])
            pose_df = pd.DataFrame(pose_arr, columns=columns)
            pose_df.insert(0, "Timestamp_ms", ts_pose)
            pose_df.to_csv(os.path.join(episode_folder, "pose.csv"), index=False)

        metadata = {
            "episode_id": str(uuid.uuid4()),
            "subject_id": self.settings.subject_id,
            "session_name": self.settings.session_name,
            "pose_name": self.settings.pose_name,
            "episode_number": episode_number,
            "duration_seconds": self.settings.episode_duration,
            "recorded_duration_seconds": 0.0 if self._recording_start is None else max(0.0, time.perf_counter() - self._recording_start),
            "sample_count_emg": len(self._emg_data),
            "sample_count_pose": len(self._pose_data),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "save_root": os.path.abspath(self.settings.save_dir),
            "session_dir": self._session_dir(),
        }
        with open(os.path.join(episode_folder, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        self._recording_start = None
        self._record_stop_event.clear()
        return episode_folder

    def _push_dashboard_status(self):
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

    def _refresh_session_counters(self):
        session_dir = self._pose_dir()
        completed = 0
        if os.path.isdir(session_dir):
            completed = len([name for name in os.listdir(session_dir) if name.startswith("ep_") and os.path.isdir(os.path.join(session_dir, name))])
        self.runtime["completed_episodes"] = completed
        self.runtime["current_episode_label"] = f"{completed} / {self.settings.episodes_per_session}"

    def _session_dir(self):
        return os.path.join(
            self.settings.save_dir,
            self.settings.subject_id,
            self.settings.session_name,
        )

    def _pose_dir(self):
        return os.path.join(self._session_dir(), self.settings.pose_name)

    def _episode_dir(self, episode_number):
        return os.path.join(self._pose_dir(), f"ep_{episode_number:04d}")

    def _set_error(self, message):
        with self._lock:
            self.runtime["mode"] = "Error"
            self.runtime["status_message"] = message
            self.runtime["last_error"] = message
            self.runtime["recording"] = False
            self.runtime["session_active"] = self.runtime["session_active"] and self.runtime["hardware_running"]
            self._push_dashboard_status()

    def _update_runtime(self, **updates):
        with self._lock:
            self.runtime.update(updates)
            self._push_dashboard_status()
