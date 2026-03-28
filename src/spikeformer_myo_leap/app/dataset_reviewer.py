"""Dataset reviewer GUI for replaying and curating recorded episodes."""

from __future__ import annotations

import os
import shutil
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from spikeformer_myo_leap.data.contracts import HAND_CONNECTIONS
from spikeformer_myo_leap.data.raw import list_episode_paths, load_episode


class HandPreviewWidget(QtWidgets.QWidget):
    """2D hand preview projected from saved Leap XYZ points."""

    VIEW_PADDING_RATIO = 0.18

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = []
        self.setMinimumHeight(260)

    def set_points(self, points):
        self._points = points or []
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#0b1220"))

        if not self._points:
            painter.setPen(QtGui.QColor("#64748b"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No pose data")
            return

        xs = [point[0] for point in self._points]
        ys = [point[1] for point in self._points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = max((max_x - min_x) * self.VIEW_PADDING_RATIO, 10.0)
        pad_y = max((max_y - min_y) * self.VIEW_PADDING_RATIO, 10.0)
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        margin = 18.0
        width = max(self.width() - 2 * margin, 1.0)
        height = max(self.height() - 2 * margin, 1.0)

        def project(point):
            x = margin + ((point[0] - min_x) / span_x) * width
            y = margin + (1.0 - ((point[1] - min_y) / span_y)) * height
            return QtCore.QPointF(x, y)

        projected = [project(point) for point in self._points]

        painter.setPen(QtGui.QPen(QtGui.QColor("#22d3ee"), 2.0))
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(projected) and end_idx < len(projected):
                painter.drawLine(projected[start_idx], projected[end_idx])

        painter.setBrush(QtGui.QColor("#fbbf24"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#f8fafc"), 1.0))
        for point in projected:
            painter.drawEllipse(point, 3.5, 3.5)


class EmgPreviewWidget(QtWidgets.QWidget):
    """Rolling EMG preview for saved episode playback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []
        self.setMinimumHeight(260)

    def set_samples(self, samples):
        self._samples = samples or []
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#0b1220"))

        if not self._samples:
            painter.setPen(QtGui.QColor("#64748b"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No EMG data")
            return

        channel_count = len(self._samples[0])
        colors = [
            "#22d3ee", "#f59e0b", "#10b981", "#e879f9",
            "#f87171", "#60a5fa", "#facc15", "#34d399",
        ]
        margin_x = 12.0
        margin_y = 12.0
        width = max(self.width() - 2 * margin_x, 1.0)
        height = max(self.height() - 2 * margin_y, 1.0)
        lane_height = height / channel_count
        sample_count = max(len(self._samples), 2)

        for channel_idx in range(channel_count):
            lane_top = margin_y + channel_idx * lane_height
            lane_center = lane_top + lane_height / 2.0
            painter.setPen(QtGui.QPen(QtGui.QColor("#1e293b"), 1.0))
            painter.drawLine(
                QtCore.QPointF(margin_x, lane_center),
                QtCore.QPointF(margin_x + width, lane_center),
            )

            channel_values = [sample[channel_idx] for sample in self._samples]
            max_abs = max(max(abs(value) for value in channel_values), 1.0)
            path = QtGui.QPainterPath()
            for sample_idx, value in enumerate(channel_values):
                x = margin_x + (sample_idx / (sample_count - 1)) * width
                normalized = value / max_abs
                y = lane_center - normalized * (lane_height * 0.35)
                if sample_idx == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QtGui.QPen(QtGui.QColor(colors[channel_idx % len(colors)]), 1.5))
            painter.drawPath(path)


class DatasetReviewerWindow(QtWidgets.QMainWindow):
    """Desktop tool for browsing, replaying, and deleting saved episodes."""

    def __init__(self):
        super().__init__()
        self.dataset_root = "datasets"
        self.episodes = []
        self.current_episode = None
        self.current_emg_samples = []
        self.current_pose_frames = []
        self.current_pose_timestamps = []
        self.current_duration_ms = 0.0
        self.playback_index = 0

        self.setWindowTitle("SpikeformerMyoLeap | Dataset Reviewer")
        self.resize(1320, 860)

        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self.advance_playback)

        self._build_ui()
        self._apply_styles()
        self.refresh_episode_list()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        self.root_edit = QtWidgets.QLineEdit(self.dataset_root)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_root)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_episode_list)
        header.addWidget(QtWidgets.QLabel("Dataset Root"))
        header.addWidget(self.root_edit, stretch=1)
        header.addWidget(browse_btn)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        layout.addLayout(body, stretch=1)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        body.addLayout(left, stretch=2)
        body.addLayout(right, stretch=3)

        self.episode_tree = QtWidgets.QTreeWidget()
        self.episode_tree.setHeaderLabels(["Subject", "Session", "Pose", "Episode"])
        self.episode_tree.itemSelectionChanged.connect(self.select_episode)
        left.addWidget(self.episode_tree, stretch=1)

        self.delete_btn = QtWidgets.QPushButton("Delete Selected Episode")
        self.delete_btn.clicked.connect(self.delete_selected_episode)
        self.delete_btn.setEnabled(False)
        left.addWidget(self.delete_btn)

        self.health_label = QtWidgets.QLabel("No episode selected")
        self.health_label.setWordWrap(True)
        self.health_label.setObjectName("HealthLabel")
        left.addWidget(self.health_label)

        right.addWidget(self._build_metadata_group())
        right.addWidget(self._build_preview_group(), stretch=1)

    def _build_metadata_group(self):
        group = QtWidgets.QGroupBox("Episode Details")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self.detail_labels = {}
        rows = [
            ("Path", "path"),
            ("Subject", "subject_id"),
            ("Session", "session_name"),
            ("Pose", "pose_name"),
            ("Episode", "episode_number"),
            ("Duration (s)", "recorded_duration_seconds"),
            ("EMG Samples", "sample_count_emg"),
            ("Pose Samples", "sample_count_pose"),
            ("EMG Hz", "effective_emg_hz"),
            ("Pose Hz", "effective_pose_hz"),
        ]
        for row, (label, key) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            value = QtWidgets.QLabel("-")
            value.setWordWrap(True)
            grid.addWidget(value, row, 1)
            self.detail_labels[key] = value
        return group

    def _build_preview_group(self):
        group = QtWidgets.QGroupBox("Replay")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(10)

        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.scrub_playback)
        self.time_label = QtWidgets.QLabel("0.00 s")
        controls.addWidget(self.play_btn)
        controls.addWidget(self.slider, stretch=1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        self.hand_preview = HandPreviewWidget()
        self.emg_preview = EmgPreviewWidget()
        layout.addWidget(self.hand_preview)
        layout.addWidget(self.emg_preview)
        return group

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget { background: #020617; color: #e5e7eb; font-size: 14px; }
            QMainWindow { background: #020617; }
            QGroupBox {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #cbd5e1;
            }
            QLineEdit, QTreeWidget {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 8px 10px;
            }
            QPushButton {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover { border-color: #22d3ee; }
            QPushButton:disabled { color: #64748b; border-color: #1e293b; }
            QLabel#HealthLabel { color: #cbd5e1; }
            """
        )

    def browse_root(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Dataset Root", self.root_edit.text() or ".")
        if selected:
            self.root_edit.setText(selected)
            self.refresh_episode_list()

    def refresh_episode_list(self):
        self.dataset_root = self.root_edit.text().strip() or "datasets"
        self.episodes = list_episode_paths(self.dataset_root)
        self.episode_tree.clear()
        for episode in self.episodes:
            parts = os.path.normpath(episode.root).split(os.sep)
            subject = parts[-4] if len(parts) >= 4 else "-"
            session = parts[-3] if len(parts) >= 3 else "-"
            pose = parts[-2] if len(parts) >= 2 else "-"
            label = parts[-1]
            item = QtWidgets.QTreeWidgetItem([subject, session, pose, label])
            item.setData(0, QtCore.Qt.UserRole, episode.root)
            self.episode_tree.addTopLevelItem(item)

        self.health_label.setText(f"Found {len(self.episodes)} complete episodes under {self.dataset_root}")
        self.clear_episode_view()

    def clear_episode_view(self):
        self.current_episode = None
        self.current_emg_samples = []
        self.current_pose_frames = []
        self.current_pose_timestamps = []
        self.current_duration_ms = 0.0
        self.playback_index = 0
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.time_label.setText("0.00 s")
        self.play_btn.setText("Play")
        self.play_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.hand_preview.set_points([])
        self.emg_preview.set_samples([])
        for label in self.detail_labels.values():
            label.setText("-")

    def select_episode(self):
        items = self.episode_tree.selectedItems()
        if not items:
            self.clear_episode_view()
            return

        episode_root = items[0].data(0, QtCore.Qt.UserRole)
        episode = next((entry for entry in self.episodes if entry.root == episode_root), None)
        if episode is None:
            return

        loaded = load_episode(episode)
        meta = loaded["meta"]
        emg_frame = loaded["emg"]
        pose_frame = loaded["pose"]

        pose_columns = [column for column in pose_frame.columns if column != "Timestamp_ms"]
        self.current_pose_timestamps = pose_frame["Timestamp_ms"].astype(float).tolist()
        self.current_pose_frames = []
        for _, row in pose_frame.iterrows():
            values = [float(row[column]) for column in pose_columns]
            self.current_pose_frames.append([values[idx : idx + 3] for idx in range(0, len(values), 3)])

        emg_columns = [column for column in emg_frame.columns if column != "Timestamp_ms"]
        self.current_emg_samples = [
            (float(row["Timestamp_ms"]), [float(row[column]) for column in emg_columns])
            for _, row in emg_frame.iterrows()
        ]

        self.current_episode = episode
        self.current_duration_ms = float(meta.get("recorded_duration_seconds", 0.0)) * 1000.0
        self.playback_index = 0
        self.slider.setRange(0, max(len(self.current_pose_frames) - 1, 0))
        self.slider.setValue(0)
        self.play_btn.setText("Play")
        self.play_btn.setEnabled(bool(self.current_pose_frames))
        self.delete_btn.setEnabled(True)

        self.detail_labels["path"].setText(episode.root)
        for key in (
            "subject_id",
            "session_name",
            "pose_name",
            "episode_number",
            "recorded_duration_seconds",
            "sample_count_emg",
            "sample_count_pose",
            "effective_emg_hz",
            "effective_pose_hz",
        ):
            self.detail_labels[key].setText(str(meta.get(key, "-")))

        emg_hz = float(meta.get("effective_emg_hz", 0.0))
        pose_hz = float(meta.get("effective_pose_hz", 0.0))
        duration_s = float(meta.get("recorded_duration_seconds", 0.0))
        health_bits = []
        if duration_s < 1.0:
            health_bits.append("Short recording")
        if emg_hz < 20.0:
            health_bits.append(f"Low EMG rate ({emg_hz:.1f} Hz)")
        if pose_hz < 20.0:
            health_bits.append(f"Low pose rate ({pose_hz:.1f} Hz)")
        if not health_bits:
            health_bits.append("Looks healthy")
        self.health_label.setText(" | ".join(health_bits))

        self.update_playback_frame(0)

    def toggle_playback(self):
        if not self.current_pose_frames:
            return
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_btn.setText("Play")
        else:
            self.playback_timer.start(40)
            self.play_btn.setText("Pause")

    def scrub_playback(self, index):
        self.playback_index = index
        self.update_playback_frame(index)

    def advance_playback(self):
        if not self.current_pose_frames or not self.playback_timer.isActive():
            return
        next_index = self.playback_index + 1
        if next_index >= len(self.current_pose_frames):
            self.playback_timer.stop()
            self.play_btn.setText("Play")
            return
        self.playback_index = next_index
        self.slider.blockSignals(True)
        self.slider.setValue(next_index)
        self.slider.blockSignals(False)
        self.update_playback_frame(next_index)

    def update_playback_frame(self, index):
        if not self.current_pose_frames:
            self.hand_preview.set_points([])
            self.emg_preview.set_samples([])
            self.time_label.setText("0.00 s")
            return

        pose_points = self.current_pose_frames[index]
        current_time_ms = self.current_pose_timestamps[index]
        emg_values = [values for timestamp, values in self.current_emg_samples if timestamp <= current_time_ms]
        self.hand_preview.set_points(pose_points)
        self.emg_preview.set_samples(emg_values[-160:])
        self.time_label.setText(f"{current_time_ms / 1000.0:.2f} s")

    def delete_selected_episode(self):
        if self.current_episode is None:
            return
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Delete Episode",
            f"Delete episode?\n\n{self.current_episode.root}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        shutil.rmtree(self.current_episode.root)
        self.refresh_episode_list()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DatasetReviewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
