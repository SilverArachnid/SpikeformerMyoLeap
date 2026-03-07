import multiprocessing as mp
import queue
import time
from collections import deque

import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = ["#fb7185", "#34d399", "#60a5fa", "#fbbf24", "#c084fc"]
EMG_COLORS = ["#22d3ee", "#38bdf8", "#60a5fa", "#818cf8", "#a78bfa", "#f472b6", "#fb7185", "#f59e0b"]
PANEL_BG = "#111827"
FIG_BG = "#020617"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
ACCENT = "#22d3ee"


def _dashboard_main(hand_queue, emg_queue, status_queue, title, show_hand, show_emg, history_size):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams.update({
        "figure.facecolor": FIG_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "text.color": TEXT,
        "font.size": 11,
    })

    hand_state = []
    emg_times = deque(maxlen=history_size)
    emg_samples = deque(maxlen=history_size)
    status = {
        "mode": "Idle",
        "recording": False,
        "pose_name": "-",
        "subject_id": "-",
        "episode_label": "-",
        "sample_count_pose": 0,
        "sample_count_emg": 0,
        "last_hand_update": None,
        "last_emg_update": None,
        "status_line": "Waiting for sensor data",
    }

    fig = plt.figure(figsize=(14, 8), facecolor=FIG_BG)
    if show_hand and show_emg:
        gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.4], width_ratios=[1.3, 1.3, 0.9], hspace=0.18, wspace=0.16)
        hand_ax = fig.add_subplot(gs[0, :2], projection="3d")
        status_ax = fig.add_subplot(gs[0, 2])
        emg_ax = fig.add_subplot(gs[1, :])
    elif show_hand:
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 0.9], wspace=0.16)
        hand_ax = fig.add_subplot(gs[0, :2], projection="3d")
        status_ax = fig.add_subplot(gs[0, 2])
        emg_ax = None
    else:
        gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 0.9, 0.01], wspace=0.16)
        emg_ax = fig.add_subplot(gs[0, 0])
        status_ax = fig.add_subplot(gs[0, 1])
        hand_ax = None

    fig.suptitle(title, fontsize=18, fontweight="bold", color=TEXT, x=0.06, ha="left")

    def style_3d_axis(ax):
        ax.set_facecolor(PANEL_BG)
        ax.xaxis.pane.set_facecolor((17 / 255, 24 / 255, 39 / 255, 1.0))
        ax.yaxis.pane.set_facecolor((17 / 255, 24 / 255, 39 / 255, 1.0))
        ax.zaxis.pane.set_facecolor((17 / 255, 24 / 255, 39 / 255, 1.0))
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("X (mm)", labelpad=10)
        ax.set_ylabel("Y (mm)", labelpad=10)
        ax.set_zlabel("Z (mm)", labelpad=10)
        ax.view_init(elev=18, azim=-68)

    def style_2d_axis(ax):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)

    if hand_ax is not None:
        style_3d_axis(hand_ax)
    if emg_ax is not None:
        style_2d_axis(emg_ax)

    def drain_queue():
        while True:
            try:
                payload, ts = hand_queue.get_nowait()
            except queue.Empty:
                break

            hand_state.clear()
            for hand_points in payload:
                hand_state.append(np.asarray(hand_points, dtype=np.float32))
            status["last_hand_update"] = ts

        emg_messages_processed = 0
        max_emg_messages_per_frame = 24
        while emg_messages_processed < max_emg_messages_per_frame:
            try:
                payload, ts = emg_queue.get_nowait()
            except queue.Empty:
                break

            emg_times.append(ts)
            emg_samples.append(np.asarray(payload, dtype=np.float32))
            status["last_emg_update"] = ts
            emg_messages_processed += 1

        while True:
            try:
                payload, ts = status_queue.get_nowait()
            except queue.Empty:
                break

            if payload == "__STOP__":
                plt.close(fig)
                break
            status.update(payload)

    def render_hand():
        if hand_ax is None:
            return

        hand_ax.clear()
        style_3d_axis(hand_ax)
        hand_ax.set_title("Hand Pose", loc="left", fontsize=13, pad=12)
        hand_ax.set_xlim(-250, 250)
        hand_ax.set_ylim(50, 420)
        hand_ax.set_zlim(-220, 220)

        if not hand_state:
            hand_ax.text2D(0.05, 0.9, "No hand tracked", transform=hand_ax.transAxes, color=MUTED, fontsize=12)
            return

        for hand_idx, points in enumerate(hand_state):
            hand_ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=26, c=ACCENT, alpha=0.95, depthshade=False)
            hand_ax.scatter(points[0, 0], points[0, 1], points[0, 2], s=60, c="#f8fafc", depthshade=False)
            for finger_idx in range(5):
                color = FINGER_COLORS[finger_idx]
                start = 1 + finger_idx * 4
                chain = [0, start, start + 1, start + 2, start + 3]
                hand_ax.plot(points[chain, 0], points[chain, 1], points[chain, 2], color=color, linewidth=2.8, alpha=0.95)

    def render_emg():
        if emg_ax is None:
            return

        emg_ax.clear()
        style_2d_axis(emg_ax)
        emg_ax.set_title("EMG Signals", loc="left", fontsize=13, pad=12)

        if len(emg_samples) < 2:
            emg_ax.text(0.03, 0.86, "Waiting for EMG stream", transform=emg_ax.transAxes, color=MUTED, fontsize=12)
            return

        samples = np.vstack(emg_samples)
        times = np.asarray(emg_times)
        time_axis = times - times[-1]
        scale = max(10.0, float(np.percentile(np.abs(samples), 98)))
        offsets = np.arange(samples.shape[1] - 1, -1, -1) * scale * 2.4

        for channel_idx in range(samples.shape[1]):
            trace = samples[:, channel_idx] + offsets[channel_idx]
            color = EMG_COLORS[channel_idx % len(EMG_COLORS)]
            emg_ax.plot(time_axis, trace, color=color, linewidth=1.8)
            emg_ax.text(
                time_axis[0] - 0.05 * max(1e-6, abs(time_axis[0])),
                offsets[channel_idx],
                f"CH{channel_idx + 1}",
                color=color,
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
            )

        emg_ax.set_xlabel("Time (s)")
        emg_ax.set_yticks([])
        emg_ax.set_xlim(time_axis[0], 0.0)
        emg_ax.set_ylim(-scale * 1.5, offsets[0] + scale * 1.5)

    def render_status():
        status_ax.clear()
        status_ax.set_facecolor(PANEL_BG)
        status_ax.set_xticks([])
        status_ax.set_yticks([])
        for spine in status_ax.spines.values():
            spine.set_color(GRID)

        recording_color = "#22c55e" if status.get("recording") else "#f59e0b"
        status_ax.text(0.08, 0.92, "Session", fontsize=13, fontweight="bold", color=TEXT, transform=status_ax.transAxes)
        status_ax.text(0.08, 0.84, "\u25cf", fontsize=18, color=recording_color, transform=status_ax.transAxes, va="center")
        status_ax.text(
            0.16,
            0.84,
            "Recording" if status.get("recording") else "Monitoring",
            fontsize=12,
            color=TEXT,
            transform=status_ax.transAxes,
            va="center",
        )

        rows = [
            ("Mode", status.get("mode", "-")),
            ("Pose", status.get("pose_name", "-")),
            ("Subject", status.get("subject_id", "-")),
            ("Episode", status.get("episode_label", "-")),
            ("Pose Samples", str(status.get("sample_count_pose", 0))),
            ("EMG Samples", str(status.get("sample_count_emg", 0))),
        ]
        y = 0.72
        for label, value in rows:
            status_ax.text(0.08, y, label, fontsize=9.5, color=MUTED, transform=status_ax.transAxes)
            status_ax.text(0.08, y - 0.055, value, fontsize=11.5, color=TEXT, transform=status_ax.transAxes, fontweight="medium")
            y -= 0.12

        hand_age = "-" if status["last_hand_update"] is None else f"{time.time() - status['last_hand_update']:.2f}s ago"
        emg_age = "-" if status["last_emg_update"] is None else f"{time.time() - status['last_emg_update']:.2f}s ago"
        status_ax.text(0.08, 0.14, "Last Hand Frame", fontsize=9.5, color=MUTED, transform=status_ax.transAxes)
        status_ax.text(0.08, 0.09, hand_age, fontsize=10.5, color=TEXT, transform=status_ax.transAxes)
        status_ax.text(0.54, 0.14, "Last EMG Sample", fontsize=9.5, color=MUTED, transform=status_ax.transAxes)
        status_ax.text(0.54, 0.09, emg_age, fontsize=10.5, color=TEXT, transform=status_ax.transAxes)
        status_ax.text(0.08, 0.02, status.get("status_line", ""), fontsize=10, color=ACCENT, transform=status_ax.transAxes)

    def update(_frame):
        drain_queue()
        render_hand()
        render_emg()
        render_status()
        return []

    animation = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
    try:
        plt.show()
    finally:
        del animation


class LocalDashboard:
    def __init__(self, title, show_hand=True, show_emg=True, history_size=320):
        self.title = title
        self.show_hand = show_hand
        self.show_emg = show_emg
        self.history_size = history_size
        self._ctx = mp.get_context("spawn")
        self._hand_queue = self._ctx.Queue(maxsize=1)
        self._emg_queue = self._ctx.Queue(maxsize=128)
        self._status_queue = self._ctx.Queue(maxsize=4)
        self._process = None

    def start(self):
        if self._process is not None and self._process.is_alive():
            return
        self._process = self._ctx.Process(
            target=_dashboard_main,
            args=(
                self._hand_queue,
                self._emg_queue,
                self._status_queue,
                self.title,
                self.show_hand,
                self.show_emg,
                self.history_size,
            ),
            daemon=True,
        )
        self._process.start()

    @staticmethod
    def _replace_latest(target_queue, payload):
        timestamp = time.time()
        while True:
            try:
                target_queue.put_nowait((payload, timestamp))
                return
            except queue.Full:
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    return

    @staticmethod
    def _append_stream(target_queue, payload):
        timestamp = time.time()
        while True:
            try:
                target_queue.put_nowait((payload, timestamp))
                return
            except queue.Full:
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    return

    def _send_stop(self):
        if self._process is None or not self._process.is_alive():
            return
        self._replace_latest(self._status_queue, "__STOP__")

    def update_hand(self, hands):
        payload = [np.asarray(hand, dtype=np.float32).tolist() for hand in hands]
        self._replace_latest(self._hand_queue, payload)

    def update_emg(self, emg):
        self._append_stream(self._emg_queue, np.asarray(emg, dtype=np.float32).tolist())

    def update_status(self, **status):
        self._replace_latest(self._status_queue, status)

    def close(self, timeout=2.0):
        if self._process is None:
            return
        self._send_stop()
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=timeout)
        self._process = None
