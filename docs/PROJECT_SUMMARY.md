# SpikeFormerMyoLeap: Project Summary and Development Log

## Goal of the Project
The primary objective of this project is to create a robust, standalone data collection and visualization pipeline for Version 2 of the SpikeFormerMyo project.

In the previous version, data collection relied on a Myo armband for 8-channel EMG data synchronized with a standard camera utilizing MediaPipe for finger tracking. In this V2 iteration, the camera and MediaPipe have been replaced by the Leap Motion Controller (Ultraleap Gemini) to capture highly accurate 3D hand tracking joint data, synchronized with preprocessed 8-channel Myo EMG data. The collected data is intended to be used for future Spikeformer model training.

## Architecture and Technical Decisions
1. Dependency and Environment Management (`uv`)
   The repository uses `uv` and `pyproject.toml` to keep the environment reproducible and self-contained.
2. Configuration (`hydra-core`)
   Recording defaults are centralized in configuration so subject IDs, episode durations, save roots, and visualization modes remain easy to change.
3. Visualization
   The repo now defaults to a local dark-mode desktop dashboard for Leap, Myo, and combined collection workflows. Rerun remains available as an optional backend.
4. Data Synchronization
   The collection flow records native-rate Leap and Myo streams with timestamps, then saves synchronized episodes with shared metadata.

## What We Discussed, Tried, and Resolved

### 1. Repository Path Correction
Development files were initially created under `SpikeFormerMyoLeap`. The repository was normalized to `SpikeformerMyoLeap` and references were updated.

### 2. LeapC Python Bindings Compilation (`setup_env.sh`)
The Leap Motion SDK bindings needed local compilation through CFFI. The final setup flow pins Python 3.10, installs required system dependencies, and exports explicit Leap header and library locations before building the bindings.

### 3. Rerun Graphics Pipeline on Linux
Native Rerun rendering was unreliable on the target Linux setup due to GPU and driver limitations. The repo now treats the local dashboard as the default visualization path, with Rerun retained only as an optional backend.

### 4. GUI Collection Workflow
The project now includes a `PySide6` collection UI for subject/session/pose setup, hardware control, and episode recording. The legacy terminal collector remains as a fallback wrapper over the same backend.

### 5. Importable `src/` Package Restructure
The codebase has been restructured into an importable package under `src/spikeformer_myo_leap/`, so collection, visualization, and future preprocessing or inference code can share the same modules.

## Final State
The repository currently provides:
1. A reproducible environment builder (`setup_env.sh`).
2. Standalone Leap and Myo preview tools.
3. A GUI-first collection workflow backed by a shared controller.
4. A legacy terminal collector that saves through the same backend.
5. Session-aware episode output with `emg.csv`, `pose.csv`, and `meta.json`.
6. An initial preprocessing/data-layer skeleton for manifest building, raw array loading, resampling, and wrist-relative pose transforms.

## Future Steps

The repository is still acquisition-first, not yet a full end-to-end port of the older `/home/pranav/pranav/SpikeFormerMyo` project.

### What is already in place
- Leap Motion hand tracking acquisition
- Myo EMG acquisition
- synchronized episode recording
- local desktop visualization
- a GUI-first collection workflow
- importable collection, visualization, and raw data IO modules
- a preprocessing config object with `xyz` as the default target mode
- raw dataset discovery and episode manifest building
- raw EMG and pose array loaders
- basic interpolation-based episode resampling
- wrist-relative pose normalization for preprocessing

### What is still missing
- sliding-window dataset generation for training
- richer preprocessing validation and reporting for incomplete episodes
- train/val/test split tooling
- additional normalization and transform options beyond wrist-relative pose
- training and evaluation pipelines
- streaming inference pipeline
- downstream robot or simulation control experiments

### Key Compatibility Finding
The new collector writes 21-joint `XYZ` pose targets and should treat `XYZ` as the default contract. The older training stack only consumed `XY`, so future preprocessing and training code should support:
1. `xyz` as the default target mode
2. `xy` as an explicit compatibility mode

### Current Risks and Known Gaps
- preprocessing is now only partially ported; it stops at preprocessed per-episode tensors and does not yet build training windows
- incomplete episodes are currently skipped implicitly rather than surfaced through a dedicated validation report
- the preprocessing layer still needs richer normalization and split-generation logic
- raw episodes now include effective per-episode sample rates, and the new preprocessing path is designed to use those values instead of assuming the old fixed camera regime

### Preprocessing Progress
The repository now includes the first preprocessing-oriented modules:
- `src/spikeformer_myo_leap/config/preprocessing.py`
- `src/spikeformer_myo_leap/data/manifest.py`
- `src/spikeformer_myo_leap/data/loaders.py`
- `src/spikeformer_myo_leap/data/preprocessing.py`
- `src/spikeformer_myo_leap/data/transforms.py`

These modules currently provide:
- manifest generation over complete recorded episodes
- `xyz`-first pose loading with optional `xy` compatibility mode
- timestamp-based resampling onto a configurable training timebase
- wrist-relative pose transformation

This is intentionally cleaner than the older project, where most preprocessing behavior was effectively collapsed into `dataset.py` and `utils.py`. The new repo is separating raw discovery, loading, transforms, and preprocessing assembly so later training and runtime inference can reuse the same interfaces cleanly.

### Recommended Implementation Order
1. finish tightening collection and dataset contracts
2. finish the preprocessing layer:
   - episode validation reporting
   - split generation
   - sliding-window dataset objects
   - additional normalization controls
3. add training and evaluation modules with configurable `xyz` or `xy` targets
4. rebuild streaming inference and downstream control on top of the shared modules
