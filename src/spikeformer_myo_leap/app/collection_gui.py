import os
import sys

from PySide6 import QtCore, QtWidgets

from spikeformer_myo_leap.collection.controller import CollectionController
from spikeformer_myo_leap.data.contracts import CollectionSettings


class CollectionMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = CollectionController()
        self.setWindowTitle("SpikeformerMyoLeap | Data Collection")
        self.resize(980, 720)

        self._build_ui()
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

        right_col.addWidget(self._build_status_group())
        right_col.addWidget(self._build_notes_group())
        right_col.addStretch(1)

        footer = QtWidgets.QLabel(
            "The dark-mode visualization dashboard opens as a separate local window when hardware is connected."
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
        form.addRow("Episode Duration (s)", self.duration_spin)
        form.addRow("Episodes Per Session", self.episodes_spin)
        form.addRow("Save Root", save_dir_row)
        return group

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
            ("Myo", "myo_connected"),
            ("Leap", "leap_connected"),
            ("Recording", "recording"),
            ("Session Active", "session_active"),
            ("Episode", "current_episode_label"),
            ("EMG Samples", "sample_count_emg"),
            ("Pose Samples", "sample_count_pose"),
            ("Last Saved", "last_saved_episode"),
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
            "1. Fill in subject, session, pose, duration, and episode count.\n"
            "2. Connect hardware to launch the dashboard and begin live monitoring.\n"
            "3. Press 'Record Next Episode' to capture one full episode.\n"
            "4. The controller saves into subject/session/pose/ep_XXXX folders.\n"
            "5. Reuse the same session to continue collecting until the target count is reached."
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
            episode_duration=float(self.duration_spin.value()),
            episodes_per_session=int(self.episodes_spin.value()),
            save_dir=self.save_dir_edit.text().strip() or "datasets",
        )

    def browse_save_dir(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Save Root", self.save_dir_edit.text() or ".")
        if selected:
            self.save_dir_edit.setText(selected)
            self.refresh_status()

    def connect_hardware(self):
        try:
            settings = self.current_settings()
            self.controller.set_settings(settings)
            self.controller.connect()
            self.refresh_status()
        except Exception as exc:
            self.show_error(f"Failed to connect hardware: {exc}")

    def disconnect_hardware(self):
        try:
            self.controller.disconnect()
            self.refresh_status()
        except Exception as exc:
            self.show_error(f"Failed to disconnect hardware: {exc}")

    def start_session(self):
        try:
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
        self.controller.set_settings(settings)

        snapshot = self.controller.get_status_snapshot()
        if snapshot["completed_episodes"] >= settings.episodes_per_session:
            self.show_error("This session already reached the configured number of episodes.")
            return

        next_episode_path = self.next_episode_path(snapshot, settings)
        if os.path.exists(next_episode_path):
            self.show_error(f"Next episode path already exists:\n{next_episode_path}")
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

    def next_episode_path(self, snapshot, settings):
        next_episode = snapshot["completed_episodes"] + 1
        return os.path.join(
            settings.save_dir,
            settings.subject_id,
            settings.session_name,
            settings.pose_name,
            f"ep_{next_episode:04d}",
        )

    def refresh_status(self):
        settings = self.current_settings()
        snapshot = self.controller.get_status_snapshot()

        self.status_labels["mode"].setText(str(snapshot["mode"]))
        self.status_labels["status_message"].setText(str(snapshot["status_message"]))
        self.status_labels["myo_connected"].setText("Connected" if snapshot["myo_connected"] else "Disconnected")
        self.status_labels["leap_connected"].setText("Connected" if snapshot["leap_connected"] else "Disconnected")
        self.status_labels["recording"].setText("Yes" if snapshot["recording"] else "No")
        self.status_labels["session_active"].setText("Yes" if snapshot["session_active"] else "No")
        self.status_labels["current_episode_label"].setText(str(snapshot["current_episode_label"]))
        self.status_labels["sample_count_emg"].setText(str(snapshot["sample_count_emg"]))
        self.status_labels["sample_count_pose"].setText(str(snapshot["sample_count_pose"]))
        self.status_labels["last_saved_episode"].setText(snapshot["last_saved_episode"] or "-")

        next_path = self.next_episode_path(snapshot, settings)
        self.path_preview.setText(f"Next episode path: {next_path}")

        warnings = []
        if os.path.exists(next_path):
            warnings.append("Next episode path already exists. Rename the session or pose before recording.")
        session_dir = os.path.join(settings.save_dir, settings.subject_id, settings.session_name)
        if os.path.isdir(session_dir):
            warnings.append("Session directory already exists. New episodes will continue numbering inside it.")
        self.warning_label.setText("\n".join(warnings))

        can_connect = not snapshot["hardware_running"]
        can_disconnect = snapshot["hardware_running"]
        can_start_session = snapshot["hardware_running"] and not snapshot["session_active"] and not snapshot["recording"]
        can_stop_session = snapshot["session_active"] and not snapshot["recording"]
        can_record = (
            snapshot["hardware_running"]
            and snapshot["session_active"]
            and snapshot["myo_connected"]
            and snapshot["leap_connected"]
            and not snapshot["recording"]
            and snapshot["completed_episodes"] < settings.episodes_per_session
            and not os.path.exists(next_path)
        )
        can_stop_record = snapshot["recording"]
        self.connect_btn.setEnabled(can_connect)
        self.disconnect_btn.setEnabled(can_disconnect)
        self.start_session_btn.setEnabled(can_start_session)
        self.stop_session_btn.setEnabled(can_stop_session)
        self.record_btn.setEnabled(can_record)
        self.stop_record_btn.setEnabled(can_stop_record)

        fields_editable = not snapshot["session_active"] and not snapshot["recording"]
        self.subject_edit.setEnabled(fields_editable)
        self.session_edit.setEnabled(fields_editable)
        self.pose_edit.setEnabled(fields_editable)
        self.duration_spin.setEnabled(fields_editable)
        self.episodes_spin.setEnabled(fields_editable)
        self.save_dir_edit.setEnabled(fields_editable)
        self.browse_btn.setEnabled(fields_editable)

    def show_error(self, message):
        QtWidgets.QMessageBox.critical(self, "Collection Error", message)

    def closeEvent(self, event):
        self.status_timer.stop()
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
