# SpikeformerMyoLeap

Version 2 of the SpikeformerMyo project. This repository contains a standalone, robust pipeline for collecting synchronized 21-joint 3D hand poses via Leap Motion (`leapc-python-bindings`) and 8-channel EMG data from the Myo armband (`pyomyo`). 

## Features
- **Standalone Environment**: Uses `uv` and `pyproject.toml` to manage dependencies seamlessly, including local CFFI compilation for the Leap SDK.
- **Hydra Configuration**: Easily adjust recording durations, subjects, and parameters via `conf/config.yaml`.
- **Professional Local Visualizers**: Dark-mode desktop dashboards for Leap-only, Myo-only, and full collection monitoring.
- **Desktop Collection UI**: A `PySide6` collection console for subject/session/pose setup, episode control, and session-aware saving.
- **Optional Rerun Path**: Rerun remains available as an optional backend, but the default workflow now avoids its Linux GPU issues.

## Quickstart

1. Clone this repository adjacent to the `leapc-python-bindings` repository.
2. Setup the environment:
```bash
./setup_env.sh
source .venv/bin/activate
```
3. (Optional) Run standalone visualizers to test hardware:
```bash
uv run visualize_leap.py
uv run visualize_myo.py
```
Both commands now default to a polished local dark-mode viewer. If you still want the older Rerun path, use:
```bash
uv run visualize_leap.py --viewer rerun
uv run visualize_myo.py --viewer rerun
```
4. Start the GUI-based data collection app:
```bash
uv run collection_gui.py
```
This is now the primary collection workflow. It provides:
- subject / session / pose configuration
- episode duration and episodes-per-session controls
- connect / disconnect hardware buttons
- one-click episode recording
- session-aware save paths
- the existing local dark-mode visualization dashboard in a separate window

5. Legacy terminal collector (fallback only):
```bash
uv run leap_myo_data_collection.py
```
*Press Spacebar to record an episode, and ESC to quit.* By default, if `visualize: true` and `visualizer_backend: "local"` are set in the config, a professional local dashboard will display:
- the tracked 3D Leap hand pose
- the rolling 8-channel EMG traces
- live session metadata such as subject, pose, episode progress, and sample counts

If you need the old Rerun path instead, set `visualizer_backend: "rerun"` in [`conf/config.yaml`](/home/pranav/pranav/SpikeformerMyoLeap/conf/config.yaml).
