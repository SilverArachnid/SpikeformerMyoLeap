# SpikeFormerMyoLeap: Project Summary and Development Log

## 🎯 Goal of the Project
The primary objective of this project is to create a robust, standalone data collection and visualization pipeline for "Version 2" of the **SpikeFormerMyo** project. 

In the previous version, data collection relied on a Myo armband for 8-channel EMG data synchronized with a standard camera utilizing MediaPipe for finger tracking. In this new V2 iteration, the camera and MediaPipe have been replaced by the **Leap Motion Controller** (Ultraleap Gemini) to capture highly accurate 3D hand tracking joint data (21 joints, 63 XYZ coordinates) synchronized perfectly with the preprocessed 8-channel Myo EMG data. The collected data is intended to be used for training Spikeformer models.

## 🏗️ Architecture & Technical Decisions
1. **Dependency & Environment Management (`uv`)**: 
   To ensure the repository is completely reproducible and self-contained, we abandoned global Python environments and utilized `uv` alongside a standard `pyproject.toml`.
2. **Configuration (`hydra-core`)**: 
   All recording parameters (episode durations, subject IDs, save paths, etc.) are centralized in `conf/config.yaml` using Hydra to maintain a professional, easily tweakable pipeline.
3. **Visualization (`rerun-sdk`)**: 
   We migrated to the Rerun SDK for dynamic 3D visualizations, enabling real-time streaming of both the Myo timeseries data and the 3D Leap hand skeletal joints.
4. **Data Synchronization**: 
   A unified collection script (`leap_myo_data_collection.py`) was written to handle headless keyboard input (via Linux `termios`) to trigger recordings seamlessly without relying on Windows-specific `msvcrt` libraries.

---

## 🛠️ What We Discussed, Tried, and Resolved

Throughout the development process, we encountered and handled several Linux-specific hardware and dependency hurdles:

### 1. Repository Path Correction
Initially, all development files were accidentally placed in a folder named `SpikeFormerMyoLeap` (capital 'F'). The original goal was to utilize the exact casing `SpikeformerMyoLeap`. We resolved this by physically moving all files via terminal and updating the import and setup scripts accordingly.

### 2. LeapC Python Bindings Compilation (`setup_env.sh`)
The Python bindings for the Leap Motion SDK (`leapc-python-bindings`) must be compiled locally using CFFI. This proved difficult inside an isolated `uv` build environment because the build backend could not locate the system's `LeapC.h` headers.
* **What we tried first:** Attempted to use PyPI dependencies directly pointing to Ultraleap's GitHub using `uv pip` and `pyproject.toml` native resolution paths.
* **The Error:** `Exception: No LeapC.h found`. The isolated build stripped environment variables and could not infer the Linux default paths reliably.
* **The Fix:** We rewrote `setup_env.sh` to explicitly lock Python to `3.10`, install the Ultraleap `apt` repositories securely via `gpg`, install the tracking service system dependencies, and manually inject `LEAPC_HEADER_OVERRIDE` and `LEAPC_LIB_OVERRIDE` environment variables before compiling the cloned `leapc-cffi` package natively.

### 3. Rerun Graphics Pipeline on Linux Wayland/Headless
When attempting to launch the standalone visualizers (`visualize_leap.py` and `visualize_myo.py`), Rerun threw `wgpu_hal::vulkan::instance` and `libEGL` errors.
* **The Cause:** Rerun natively attempts to spawn a heavy 3D GPU window. On certain Linux setups (especially under Wayland or without correct DRM extensions), this fails to allocate the GUI context.
* **The Fix:** We implemented a `--web` / `web_viewer: true` fallback across all visualizer and collection scripts. This replaces the deprecated `rr.serve()` API with the up-to-date `rr.serve_web_viewer()` API, hosting the 3D viewport internally on a local web server (usually port `9090`). This fully bypasses the native window manager and driver stack, allowing the visualizations to render comfortably in the user's web browser.

---

## 🚀 Final State
The repository now successfully offers:
1. A reproducible environment builder (`setup_env.sh`).
2. Standalone streaming tests (`visualize_myo.py --web` and `visualize_leap.py --web`).
3. A robust, trigger-based data collector generating exactly formatted `pose.csv` and `emg.csv` matching the downstream Spikeformer model requirements.\

---

## 🔭 Future Steps

The repository is currently an acquisition-first rewrite of SpikeFormerMyo, not yet a full end-to-end port of the older project. The current codebase cleanly handles:
- Leap Motion hand tracking acquisition
- Myo EMG acquisition
- synchronized episode recording
- live visualization through Rerun

What is still missing, compared with the older `/home/pranav/pranav/SpikeFormerMyo` repository, is the full modeling and evaluation stack:
- dataset loading and resampling utilities
- pose normalization and preprocessing helpers
- Spikeformer / spiking CNN / evaluation scripts
- realtime inference playback tools
- robot / PyBullet control experiments

### Key Compatibility Finding
The new collector already writes `pose.csv` using the same 21 landmark naming scheme as the older project, and it records full `X/Y/Z` coordinates for all 21 joints. However, the older training code only consumes `X/Y` coordinates, even though the old raw `pose.csv` files also contained `Z`.

This means the current collector is broadly file-format compatible with the old project, but the learning stack is not yet Leap-native. A deliberate modeling decision is still required:
1. Keep the older 2D target contract and ignore `Z` for now.
2. Upgrade the full data/model pipeline to predict all 63 pose values (`21 x 3`).
3. Replace raw Cartesian targets with a more useful representation such as joint angles or wrist-relative kinematic features.

### Current Risks / Known Gaps
- The project is still collection-first; preprocessing, dataset assembly, and training code have not yet been ported into this repository.
- The raw collection format is now much cleaner, but the future preprocessing layer still needs to decide how to use `XYZ` by default while retaining an optional `XY` training mode.
- The `datasets/` directory is still empty in this repository, so the new collector has not yet been validated against a real accumulated dataset inside this repo.

### Visualization Progress
The project has now moved away from relying on Rerun as the primary visualization path on Linux. In practice, native Rerun rendering was not reliable on the target machine due to GPU / driver limitations, and the browser workflow was not ideal for local collection sessions.

To address this, the repository now includes a shared local visualization layer that provides:
- a dark-mode desktop Leap hand visualizer
- a dark-mode desktop Myo EMG visualizer
- a unified local collection dashboard showing both the 3D hand pose and 8-channel EMG traces alongside session state

Rerun is still kept as an optional backend, but the intended default workflow is now the local professional dashboard path.

### Collection UI Progress
The repository is now also moving toward a proper GUI-first collection workflow instead of relying only on the terminal-triggered script.

The current direction is:
- a reusable collection controller that owns Leap + Myo hardware lifecycle, recording state, and episode saving
- a `PySide6` desktop collection console for subject/session/pose setup and episode control
- the existing dark-mode local dashboard kept as the live visualization companion window

Once this GUI path is stable, the older terminal collection flow can be treated as a legacy fallback rather than the primary user-facing workflow.

### Restructure Progress
The codebase is also now being restructured into an importable `src/` package so that future preprocessing, training, and streaming inference work can reuse the same collection and visualization logic instead of duplicating it in standalone scripts.

The current package direction is:
- `src/spikeformer_myo_leap/collection/`
- `src/spikeformer_myo_leap/visualization/`
- `src/spikeformer_myo_leap/data/`
- `src/spikeformer_myo_leap/app/`

Top-level scripts are being kept as thin wrappers so current commands remain stable while the internals become properly modular.

### Recommended Implementation Order
To rebuild this project neatly around Leap, the next steps should follow this order:
1. Fix current collector and data-management bugs first.
2. Define the final dataset contract clearly:
   - folder naming
   - metadata schema
   - overwrite protection
   - target representation (`XY`, `XYZ`, or angles)
3. Port the old shared pipeline pieces into this repository:
   - `dataset.py`
   - `utils.py`
   - `models.py`
   - `train.py`
   - `evaluate.py`
4. Refactor them so they work cleanly with Leap-based pose targets.
5. Rebuild realtime inference and downstream control/visualization only after the training contract is stable.
