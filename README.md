# SpikeformerMyoLeap

Version 2 of the SpikeformerMyo project. This repository contains a standalone, robust pipeline for collecting synchronized 21-joint 3D hand poses via Leap Motion (`leapc-python-bindings`) and 8-channel EMG data from the Myo armband (`pyomyo`). 

## Features
- **Standalone Environment**: Uses `uv` and `pyproject.toml` to manage dependencies seamlessly, including local CFFI compilation for the Leap SDK.
- **Hydra Configuration**: Easily adjust recording durations, subjects, and parameters via `conf/config.yaml`.
- **Professional Local Visualizers**: Dark-mode desktop dashboards for Leap-only, Myo-only, and full collection monitoring.
- **Desktop Collection UI**: A `PySide6` collection console for subject/session/pose setup, session control, and guided episode recording.
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
Both commands now default to the local dark-mode viewer. If you still want the Rerun path, use:
```bash
uv run visualize_leap.py --viewer rerun
uv run visualize_myo.py --viewer rerun
```
4. Start the GUI-based collection app:
```bash
uv run collection_gui.py
```
This is now the primary collection workflow. It provides:
- subject / session / pose configuration
- episode duration and episodes-per-session controls
- connect / disconnect hardware
- start / stop session
- record next episode
- stop recording early
- the existing local dashboard in a separate window for hand + EMG monitoring

5. Legacy terminal collector (fallback only):
```bash
uv run leap_myo_data_collection.py
```
*Press Spacebar to record an episode, `s` to stop early and save, and ESC or `q` to quit.*

## Data Layout

Both the GUI collector and the legacy terminal collector now save through the same backend and use the same folder structure:

```text
datasets/<subject_id>/<session_name>/<pose_name>/ep_0001/
```

Each episode contains:
- `emg.csv`
- `pose.csv`
- `meta.json`

`meta.json` includes the effective per-episode sampling rates:
- `effective_emg_hz`
- `effective_pose_hz`

These are derived from the actual captured sample counts and recorded duration, and should be used later during data cleaning and resampling rather than assuming a fixed acquisition rate.
