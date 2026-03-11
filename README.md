# SpikeformerMyoLeap

Version 2 of the SpikeformerMyo project. This repository contains a standalone, robust pipeline for collecting synchronized 21-joint 3D hand poses via Leap Motion (`leapc-python-bindings`) and 8-channel EMG data from the Myo armband (`pyomyo`).

## Features
- **Standalone Environment**: Uses `uv` and `pyproject.toml` to manage dependencies, including local CFFI compilation for the Leap SDK.
- **Hydra Configuration**: Collection defaults live in `conf/config.yaml`.
- **Importable Package Layout**: Core collection, visualization, and data-IO logic now live under `src/spikeformer_myo_leap/` instead of only in top-level scripts.
- **Package Entry Points**: Runnable script entry points now live under `src/spikeformer_myo_leap/scripts/`, while top-level scripts remain compatibility wrappers.
- **Professional Local Visualizers**: Dark-mode desktop dashboards for Leap-only, Myo-only, and full collection monitoring.
- **Desktop Collection UI**: A `PySide6` collection console for subject/session/pose setup, session control, and guided episode recording.
- **Optional Rerun Path**: Rerun remains available as an optional backend, but the default workflow now avoids its Linux GPU issues.

See [Project Summary](docs/PROJECT_SUMMARY.md) for the longer architecture and roadmap notes.

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
Both commands default to the local dark-mode viewer. If you still want the Rerun path, use:
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
- connect / disconnect hardware buttons
- start / stop session
- record next episode
- stop recording early
- session-aware save paths
- the existing local dark-mode visualization dashboard in a separate window

5. Legacy terminal collector (fallback only):
```bash
uv run leap_myo_data_collection.py
```
Press `Space` to record the next episode, `s` to stop early and save, and `Esc` or `q` to quit.

If `visualize: true` and `visualizer_backend: "local"` are enabled, the legacy collector uses the same local dashboard for:
- tracked 3D Leap hand pose
- rolling 8-channel EMG traces
- live session metadata such as subject, pose, episode progress, and sample counts

## Data Layout

Both the GUI collector and the legacy terminal collector save through the same backend and use the same folder structure:

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

## Code Layout

The current working code is now organized as an importable package:

```text
src/spikeformer_myo_leap/
  app/
  collection/
  data/
  scripts/
  visualization/
```

Current responsibilities:
- `app/`: desktop GUI entry logic
- `collection/`: hardware lifecycle, recording controller, terminal collector wrapper
- `data/`: shared contracts, save/load helpers, raw episode discovery
- `scripts/`: package-level runnable entry points
- `visualization/`: local dashboard and optional Rerun viewers

The top-level scripts are kept as thin wrappers so existing commands still work.

## Preprocessing Readiness

This PR does not add preprocessing or model training yet, but it prepares the codebase for that next step by introducing:
- a shared data contract for collection settings and landmark naming
- centralized episode save/load helpers
- raw dataset discovery utilities via `spikeformer_myo_leap.data.raw`

That means the next preprocessing/training work can be built on stable importable interfaces instead of adding more logic into top-level scripts.
