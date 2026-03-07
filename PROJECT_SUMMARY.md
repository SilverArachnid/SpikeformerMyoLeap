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
3. A robust, trigger-based data collector generating exactly formatted `pose.csv` and `emg.csv` matching the downstream Spikeformer model requirements.
