import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from spikeformer_myo_leap.collection.worker import CollectionWorkerClient
from spikeformer_myo_leap.data.contracts import CollectionSettings, HAND_CONNECTIONS


class HardwareActionThread(QtCore.QThread):
    """Run a blocking controller action off the Qt UI thread."""

    failed = QtCore.Signal(str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self._action = action

    def run(self):
        try:
            self._action()
        except Exception as exc:
            self.failed.emit(str(exc))


class HandPreviewWidget(QtWidgets.QWidget):
    """Lightweight 2D hand preview projected from Leap XYZ points."""

    VIEW_PADDING_RATIO = 0.18

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = []
        self.setMinimumHeight(220)

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
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Leap hand preview")
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
    """Lightweight rolling 8-channel EMG preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []
        self.setMinimumHeight(220)

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
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No Myo EMG preview")
            return

        channel_count = len(self._samples[0])
        if channel_count == 0:
            return
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


class CollectionMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = CollectionWorkerClient()
        self.settings_store = QtCore.QSettings("SpikeformerMyoLeap", "CollectionGUI")
        self.hardware_action_thread = None
        self.hardware_action_name = ""
        self.setWindowTitle("SpikeformerMyoLeap | Data Collection")
        self.resize(980, 720)

        self._build_ui()
        self._load_persisted_fields()
        self._connect_field_persistence()
        self._apply_styles()

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.start(150)
        self.refresh_status()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QtWidgets.QLabel("Data Collection Console")
        header.setObjectName("Header")
        subheader = QtWidgets.QLabel("Configure a collection session, connect hardware, and record synchronized Leap + Myo episodes.")
        subheader.setObjectName("Subheader")

        layout.addWidget(header)
        layout.addWidget(subheader)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(18)
        layout.addLayout(content, stretch=1)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(16)
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(16)
        content.addLayout(left_col, stretch=3)
        content.addLayout(right_col, stretch=2)

        left_col.addWidget(self._build_session_group())
        left_col.addWidget(self._build_controls_group())
        left_col.addStretch(1)

        right_col.addWidget(self._build_preview_group())
        right_col.addWidget(self._build_status_group())
        right_col.addWidget(self._build_notes_group())
        right_col.addStretch(1)

        footer = QtWidgets.QLabel(
            "The collection GUI now runs without spawning the separate dashboard window."
        )
        footer.setObjectName("Footnote")
        layout.addWidget(footer)

    def _build_session_group(self):
        group = QtWidgets.QGroupBox("Session Setup")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)

        self.subject_edit = QtWidgets.QLineEdit("user_1")
        self.session_edit = QtWidgets.QLineEdit("session_1")
        self.pose_edit = QtWidgets.QLineEdit("test_pose")
        self.recording_mode_combo = QtWidgets.QComboBox()
        self.recording_mode_combo.addItem("Episodic", "episodic")
        self.recording_mode_combo.addItem("Continuous Block", "continuous")
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 120.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setValue(5.0)

        self.episodes_spin = QtWidgets.QSpinBox()
        self.episodes_spin.setRange(1, 10000)
        self.episodes_spin.setValue(20)

        save_dir_row = QtWidgets.QHBoxLayout()
        self.save_dir_edit = QtWidgets.QLineEdit("datasets")
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_save_dir)
        save_dir_row.addWidget(self.save_dir_edit, stretch=1)
        save_dir_row.addWidget(self.browse_btn)

        form.addRow("Subject ID", self.subject_edit)
        form.addRow("Session Name", self.session_edit)
        form.addRow("Pose Name", self.pose_edit)
        form.addRow("Recording Mode", self.recording_mode_combo)
        form.addRow("Episode Duration (s)", self.duration_spin)
        form.addRow("Episodes Per Session", self.episodes_spin)
        form.addRow("Save Root", save_dir_row)
        return group

    def _build_preview_group(self):
        group = QtWidgets.QGroupBox("Live Preview")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(10)

        hand_label = QtWidgets.QLabel("Leap Hand")
        hand_label.setObjectName("KeyLabel")
        self.hand_preview = HandPreviewWidget()

        emg_label = QtWidgets.QLabel("Myo EMG")
        emg_label.setObjectName("KeyLabel")
        self.emg_preview = EmgPreviewWidget()

        layout.addWidget(hand_label)
        layout.addWidget(self.hand_preview)
        layout.addWidget(emg_label)
        layout.addWidget(self.emg_preview)
        return group

    def _load_persisted_fields(self):
        self.subject_edit.setText(str(self.settings_store.value("subject_id", self.subject_edit.text())))
        self.session_edit.setText(str(self.settings_store.value("session_name", self.session_edit.text())))
        self.pose_edit.setText(str(self.settings_store.value("pose_name", self.pose_edit.text())))
        saved_mode = str(self.settings_store.value("recording_mode", self.recording_mode_combo.currentData()))
        mode_index = self.recording_mode_combo.findData(saved_mode)
        self.recording_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self.duration_spin.setValue(float(self.settings_store.value("episode_duration", self.duration_spin.value())))
        self.episodes_spin.setValue(int(self.settings_store.value("episodes_per_session", self.episodes_spin.value())))
        self.save_dir_edit.setText(str(self.settings_store.value("save_dir", self.save_dir_edit.text())))

    def _connect_field_persistence(self):
        self.subject_edit.editingFinished.connect(self._persist_fields)
        self.session_edit.editingFinished.connect(self._persist_fields)
        self.pose_edit.editingFinished.connect(self._persist_fields)
        self.recording_mode_combo.currentIndexChanged.connect(lambda _value: self._persist_fields())
        self.duration_spin.valueChanged.connect(lambda _value: self._persist_fields())
        self.episodes_spin.valueChanged.connect(lambda _value: self._persist_fields())
        self.save_dir_edit.editingFinished.connect(self._persist_fields)

    def _persist_fields(self):
        settings = self.current_settings()
        self.settings_store.setValue("subject_id", settings.subject_id)
        self.settings_store.setValue("session_name", settings.session_name)
        self.settings_store.setValue("pose_name", settings.pose_name)
        self.settings_store.setValue("recording_mode", settings.recording_mode)
        self.settings_store.setValue("episode_duration", settings.episode_duration)
        self.settings_store.setValue("episodes_per_session", settings.episodes_per_session)
        self.settings_store.setValue("save_dir", settings.save_dir)
        self.settings_store.sync()

    def _build_controls_group(self):
        group = QtWidgets.QGroupBox("Controls")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(12)

        top_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect Hardware")
        self.connect_btn.clicked.connect(self.connect_hardware)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_hardware)
        top_row.addWidget(self.connect_btn)
        top_row.addWidget(self.disconnect_btn)

        session_row = QtWidgets.QHBoxLayout()
        self.start_session_btn = QtWidgets.QPushButton("Start Session")
        self.start_session_btn.clicked.connect(self.start_session)
        self.stop_session_btn = QtWidgets.QPushButton("Stop Session")
        self.stop_session_btn.clicked.connect(self.stop_session)
        session_row.addWidget(self.start_session_btn)
        session_row.addWidget(self.stop_session_btn)

        record_row = QtWidgets.QHBoxLayout()
        self.record_btn = QtWidgets.QPushButton("Record Next Episode")
        self.record_btn.clicked.connect(self.start_recording)
        self.stop_record_btn = QtWidgets.QPushButton("Stop Recording Early")
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.refresh_btn = QtWidgets.QPushButton("Refresh Session State")
        self.refresh_btn.clicked.connect(self.refresh_status)
        record_row.addWidget(self.record_btn)
        record_row.addWidget(self.stop_record_btn)
        record_row.addWidget(self.refresh_btn)

        self.path_preview = QtWidgets.QLabel("")
        self.path_preview.setObjectName("PreviewLabel")
        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setObjectName("WarningLabel")
        self.warning_label.setWordWrap(True)

        layout.addLayout(top_row)
        layout.addLayout(session_row)
        layout.addLayout(record_row)
        layout.addWidget(self.path_preview)
        layout.addWidget(self.warning_label)
        return group

    def _build_status_group(self):
        group = QtWidgets.QGroupBox("Live Status")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.status_labels = {}
        rows = [
            ("Mode", "mode"),
            ("Message", "status_message"),
            ("Recording Mode", "recording_mode"),
            ("Myo", "myo_connected"),
            ("Myo Stream", "myo_streaming"),
            ("Leap", "leap_connected"),
            ("Leap Stream", "leap_streaming"),
            ("Recording", "recording"),
            ("Session Active", "session_active"),
            ("Episode", "current_episode_label"),
            ("EMG Samples", "sample_count_emg"),
            ("Pose Samples", "sample_count_pose"),
            ("Last Saved", "last_saved_episode"),
            ("Last Aborted", "last_aborted_episode"),
        ]

        for row_idx, (label, key) in enumerate(rows):
            key_label = QtWidgets.QLabel(label)
            key_label.setObjectName("KeyLabel")
            value_label = QtWidgets.QLabel("-")
            value_label.setObjectName("ValueLabel")
            value_label.setWordWrap(True)
            grid.addWidget(key_label, row_idx, 0)
            grid.addWidget(value_label, row_idx, 1)
            self.status_labels[key] = value_label

        return group

    def _build_notes_group(self):
        group = QtWidgets.QGroupBox("Workflow")
        layout = QtWidgets.QVBoxLayout(group)

        notes = QtWidgets.QLabel(
            "1. Fill in subject, session, pose, recording mode, duration, and episode count.\n"
            "2. Connect hardware to begin live monitoring.\n"
            "3. In Episodic mode, 'Record Next Episode' captures and saves one episode immediately.\n"
            "4. In Continuous Block mode, one recording fills the remaining fixed-duration episode slots and saves them at the end.\n"
            "5. The controller saves into subject/session/pose/ep_XXXX folders in both modes.\n"
            "6. Reuse the same session to continue collecting until the target count is reached."
        )
        notes.setWordWrap(True)
        notes.setObjectName("Notes")
        layout.addWidget(notes)
        return group

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #020617;
                color: #e5e7eb;
                font-size: 14px;
            }
            QMainWindow {
                background: #020617;
            }
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
                padding: 0 6px 0 6px;
                color: #cbd5e1;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
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
            QPushButton:hover {
                border-color: #22d3ee;
            }
            QPushButton:disabled {
                color: #64748b;
                border-color: #1e293b;
            }
            QLabel#Header {
                font-size: 28px;
                font-weight: 700;
                color: #f8fafc;
            }
            QLabel#Subheader {
                color: #94a3b8;
                font-size: 15px;
            }
            QLabel#Footnote, QLabel#Notes {
                color: #94a3b8;
            }
            QLabel#PreviewLabel {
                color: #22d3ee;
                font-family: monospace;
            }
            QLabel#WarningLabel {
                color: #fbbf24;
            }
            QLabel#KeyLabel {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#ValueLabel {
                color: #f8fafc;
                font-size: 14px;
                font-weight: 600;
            }
            """
        )

    def current_settings(self):
        return CollectionSettings(
            subject_id=self.subject_edit.text().strip() or "user_1",
            session_name=self.session_edit.text().strip() or "session_1",
            pose_name=self.pose_edit.text().strip() or "test_pose",
            recording_mode=str(self.recording_mode_combo.currentData() or "episodic"),
            episode_duration=float(self.duration_spin.value()),
            episodes_per_session=int(self.episodes_spin.value()),
            save_dir=self.save_dir_edit.text().strip() or "datasets",
        )

    def browse_save_dir(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Save Root", self.save_dir_edit.text() or ".")
        if selected:
            self.save_dir_edit.setText(selected)
            self._persist_fields()
            self.refresh_status()

    def connect_hardware(self):
        settings = self.current_settings()
        self._persist_fields()

        def action():
            self.controller.set_settings(settings)
            self.controller.connect()

        self._start_hardware_action("Connecting hardware...", action, "Failed to connect hardware")

    def disconnect_hardware(self):
        def action():
            try:
                self.controller.disconnect()
            except Exception:
                self.controller.reset()

        self._start_hardware_action(
            "Disconnecting hardware...",
            action,
            "Failed to disconnect hardware",
        )

    def start_session(self):
        try:
            self._persist_fields()
            self.controller.start_session(self.current_settings())
            self.refresh_status()
        except Exception as exc:
            self.show_error(str(exc))

    def stop_session(self):
        try:
            self.controller.stop_session()
            self.refresh_status()
        except Exception as exc:
            self.show_error(str(exc))

    def start_recording(self):
        settings = self.current_settings()
        self._persist_fields()
        self.controller.set_settings(settings)

        snapshot = self.controller.get_status_snapshot()
        if snapshot["completed_episodes"] >= settings.episodes_per_session:
            self.show_error("This session already reached the configured number of episodes.")
            return

        planned_paths = self.planned_episode_paths(snapshot, settings)
        existing_paths = [path for path in planned_paths if os.path.exists(path)]
        if existing_paths:
            self.show_error(
                "One or more output episode paths already exist:\n"
                + "\n".join(existing_paths[:5])
                + ("\n..." if len(existing_paths) > 5 else "")
            )
            return

        try:
            self.controller.start_episode()
            self.refresh_status()
        except Exception as exc:
            self.show_error(str(exc))

    def stop_recording(self):
        try:
            self.controller.stop_episode()
            self.refresh_status()
        except Exception as exc:
            self.show_error(str(exc))

    def planned_episode_paths(self, snapshot, settings):
        next_episode = snapshot["completed_episodes"] + 1
        count = 1
        if settings.recording_mode == "continuous":
            count = max(0, settings.episodes_per_session - snapshot["completed_episodes"])
        return [
            os.path.join(
                settings.save_dir,
                settings.subject_id,
                settings.session_name,
                settings.pose_name,
                f"ep_{episode_number:04d}",
            )
            for episode_number in range(next_episode, next_episode + count)
        ]

    def refresh_status(self):
        settings = self.current_settings()
        snapshot = self.controller.get_status_snapshot()
        preview = self.controller.get_preview_snapshot()
        recording_mode = str(snapshot.get("recording_mode", settings.recording_mode))

        self.status_labels["mode"].setText(str(snapshot["mode"]))
        self.status_labels["status_message"].setText(str(snapshot["status_message"]))
        self.status_labels["recording_mode"].setText(
            "Continuous Block" if recording_mode == "continuous" else "Episodic"
        )
        self.status_labels["myo_connected"].setText("Connected" if snapshot["myo_connected"] else "Disconnected")
        self.status_labels["myo_streaming"].setText("Healthy" if snapshot["myo_streaming"] else "Waiting")
        self.status_labels["leap_connected"].setText("Connected" if snapshot["leap_connected"] else "Disconnected")
        self.status_labels["leap_streaming"].setText("Healthy" if snapshot["leap_streaming"] else "Waiting")
        self.status_labels["recording"].setText("Yes" if snapshot["recording"] else "No")
        self.status_labels["session_active"].setText("Yes" if snapshot["session_active"] else "No")
        self.status_labels["current_episode_label"].setText(str(snapshot["current_episode_label"]))
        self.status_labels["sample_count_emg"].setText(str(snapshot["sample_count_emg"]))
        self.status_labels["sample_count_pose"].setText(str(snapshot["sample_count_pose"]))
        self.status_labels["last_saved_episode"].setText(snapshot["last_saved_episode"] or "-")
        self.status_labels["last_aborted_episode"].setText(snapshot["last_aborted_episode"] or "-")
        self.hand_preview.set_points(preview.get("hand_points", []))
        self.emg_preview.set_samples(preview.get("emg_window", []))

        planned_paths = self.planned_episode_paths(snapshot, settings)
        if not planned_paths:
            self.path_preview.setText("No remaining episode slots in this session.")
        elif len(planned_paths) == 1:
            self.path_preview.setText(f"Next episode path: {planned_paths[0]}")
        else:
            self.path_preview.setText(
                f"Continuous block will save: {planned_paths[0]} ... {planned_paths[-1]}"
            )

        warnings = []
        existing_paths = [path for path in planned_paths if os.path.exists(path)]
        if existing_paths:
            warnings.append("One or more planned episode paths already exist. Rename the session or pose before recording.")
        session_dir = os.path.join(settings.save_dir, settings.subject_id, settings.session_name)
        if os.path.isdir(session_dir):
            warnings.append("Session directory already exists. New episodes will continue numbering inside it.")
        if snapshot["hardware_running"] and (not snapshot["myo_streaming"] or not snapshot["leap_streaming"]):
            warnings.append("Waiting for healthy data flow from both sensors before recording can resume.")
        if snapshot["last_aborted_episode"]:
            warnings.append(f"Most recent recording was aborted: {snapshot['last_aborted_episode']}.")
        if self.hardware_action_name:
            warnings.append(self.hardware_action_name)
        self.warning_label.setText("\n".join(warnings))

        action_running = self.hardware_action_thread is not None and self.hardware_action_thread.isRunning()

        can_connect = not action_running and not snapshot["hardware_running"]
        can_disconnect = not action_running and snapshot["hardware_running"]
        can_start_session = (
            not action_running and snapshot["hardware_running"] and not snapshot["session_active"] and not snapshot["recording"]
        )
        can_stop_session = not action_running and snapshot["session_active"] and not snapshot["recording"]
        can_record = (
            not action_running
            and snapshot["hardware_running"]
            and snapshot["session_active"]
            and snapshot["myo_connected"]
            and snapshot["leap_connected"]
            and snapshot["myo_streaming"]
            and snapshot["leap_streaming"]
            and not snapshot["recording"]
            and snapshot["completed_episodes"] < settings.episodes_per_session
            and not existing_paths
        )
        can_stop_record = not action_running and snapshot["recording"]
        self.connect_btn.setEnabled(can_connect)
        self.disconnect_btn.setEnabled(can_disconnect)
        self.start_session_btn.setEnabled(can_start_session)
        self.stop_session_btn.setEnabled(can_stop_session)
        self.record_btn.setEnabled(can_record)
        self.stop_record_btn.setEnabled(can_stop_record)
        if settings.recording_mode == "continuous":
            self.record_btn.setText("Record Remaining Block")
            self.stop_record_btn.setText("Stop Block and Save Full Segments")
        else:
            self.record_btn.setText("Record Next Episode")
            self.stop_record_btn.setText("Stop Recording Early")

        fields_editable = not action_running and not snapshot["session_active"] and not snapshot["recording"]
        self.subject_edit.setEnabled(fields_editable)
        self.session_edit.setEnabled(fields_editable)
        self.pose_edit.setEnabled(fields_editable)
        self.recording_mode_combo.setEnabled(fields_editable)
        self.duration_spin.setEnabled(fields_editable)
        self.episodes_spin.setEnabled(fields_editable)
        self.save_dir_edit.setEnabled(fields_editable)
        self.browse_btn.setEnabled(fields_editable)

    def show_error(self, message):
        QtWidgets.QMessageBox.critical(self, "Collection Error", message)

    def _start_hardware_action(self, status_text, action, error_prefix):
        if self.hardware_action_thread is not None and self.hardware_action_thread.isRunning():
            return

        self.hardware_action_name = status_text
        thread = HardwareActionThread(action, self)
        self.hardware_action_thread = thread
        thread.finished.connect(self._on_hardware_action_finished)
        thread.failed.connect(lambda message: self._on_hardware_action_failed(error_prefix, message))
        thread.start()
        self.refresh_status()

    def _on_hardware_action_finished(self):
        self.hardware_action_name = ""
        self.hardware_action_thread = None
        self.refresh_status()

    def _on_hardware_action_failed(self, prefix, message):
        self.hardware_action_name = ""
        self.hardware_action_thread = None
        self.refresh_status()
        self.show_error(f"{prefix}: {message}")

    def closeEvent(self, event):
        self.status_timer.stop()
        self._persist_fields()
        self.controller.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CollectionMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
