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
   The repo now defaults to a local dark-mode desktop dashboard for standalone visualization workflows. The collection GUI itself now uses embedded native previews instead of spawning a separate dashboard window.
4. Data Synchronization
   The collection flow records native-rate Leap and Myo streams with timestamps, then saves synchronized episodes with shared metadata.
5. GUI / Hardware Isolation
   The collection GUI now talks to a separate worker process so hardware/control failures do not directly freeze the Qt window.

## What We Discussed, Tried, and Resolved

### 1. Repository Path Correction
Development files were initially created under `SpikeFormerMyoLeap`. The repository was normalized to `SpikeformerMyoLeap` and references were updated.

### 2. LeapC Python Bindings Compilation (`setup_env.sh`)
The Leap Motion SDK bindings needed local compilation through CFFI. The final setup flow pins Python 3.10, installs required system dependencies, and exports explicit Leap header and library locations before building the bindings.

### 3. Rerun Graphics Pipeline on Linux
Native Rerun rendering was unreliable on the target Linux setup due to GPU and driver limitations. The repo now treats the local dashboard as the default visualization path, with Rerun retained only as an optional backend.

### 4. GUI Collection Workflow
The project now includes a `PySide6` collection UI for subject/session/pose setup, hardware control, and episode recording. That GUI has since been refactored to:
- run hardware/control in a worker process
- show embedded Leap-hand and Myo-EMG previews
- avoid spawning a separate collection-time dashboard window

The legacy terminal collector remains as a fallback wrapper over the same core controller logic.

### 5. Importable `src/` Package Restructure
The codebase has been restructured into an importable package under `src/spikeformer_myo_leap/`, so collection, visualization, and future preprocessing or inference code can share the same modules.

### 6. Collector Recovery and Inter-Episode Stability
The collection controller was further hardened after repeated inter-episode failures. The notable changes were:
- explicit stream-health monitoring for Myo and Leap
- explicit episode finalization state
- save-time validation for incomplete/bad captures
- continuous stream buffering so episodes are sliced from ongoing streams
- Myo cleanup and reconnect handling after device/port faults

This reduced lifecycle bugs around episode stop/save boundaries and made recovery behavior clearer.

### 7. Dataset Reviewer Tool
A separate desktop reviewer was added so saved episodes can be replayed and curated after collection. It supports:
- browsing episodes under a dataset root
- replaying Leap pose and rolling EMG
- checking simple per-episode health indicators
- deleting bad episodes

This is useful for post-collection QA before preprocessing or training.

### 8. Training and Evaluation Foundations
The repo is no longer only a collection project. It now includes:
- separate model modules for Spikeformer, Transformer, CNN-LSTM, CNN, and spiking CNN
- Hydra-backed training and evaluation entry points
- episode-level train/val splitting
- full-episode qualitative validation support

This is still foundation-level work, but it is enough to run baseline experiments on the new Leap+Myo dataset format.

## Final State
The repository currently provides:
1. A reproducible environment builder (`setup_env.sh`).
2. Standalone Leap and Myo preview tools.
3. A GUI-first collection workflow backed by a shared controller.
4. A legacy terminal collector that saves through the same backend.
5. Session-aware episode output with `emg.csv`, `pose.csv`, and `meta.json`.
6. A worker-backed collection GUI with embedded live previews and improved recovery behavior.
7. A dataset reviewer GUI for replaying and curating saved episodes.
8. An initial preprocessing/data-layer skeleton for manifest building, raw array loading, resampling, and wrist-relative pose transforms.
9. A packaged training/evaluation foundation with separated model modules.

## Future Steps

The repository is still acquisition-first, not yet a full end-to-end port of the older `/home/pranav/pranav/SpikeFormerMyo` project.

### What is already in place
- Leap Motion hand tracking acquisition
- Myo EMG acquisition
- synchronized episode recording
- local desktop visualization
- a GUI-first collection workflow
- a worker-backed GUI collection path with embedded preview panels
- a dataset reviewer GUI for saved-episode inspection and deletion
- importable collection, visualization, and raw data IO modules
- a preprocessing config object with `xyz` as the default target mode
- raw dataset discovery and episode manifest building
- raw EMG and pose array loaders
- basic interpolation-based episode resampling
- wrist-relative pose normalization for preprocessing
- packaged training/evaluation entry points
- separated model-family implementations

### What is still missing
- richer preprocessing validation and reporting for incomplete episodes
- train/val/test split tooling
- additional normalization and transform options beyond wrist-relative pose
- streaming inference pipeline
- downstream robot or simulation control experiments
- more robust long-run handling of Myo transport faults
- richer dataset-review health metrics and batch curation tools

### Key Compatibility Finding
The new collector writes 21-joint `XYZ` pose targets and should treat `XYZ` as the default contract. The older training stack only consumed `XY`, so future preprocessing and training code should support:
1. `xyz` as the default target mode
2. `xy` as an explicit compatibility mode

### Current Risks and Known Gaps
- preprocessing is now only partially ported; it stops at preprocessed per-episode tensors and does not yet build training windows
- incomplete episodes are currently skipped implicitly rather than surfaced through a dedicated validation report
- the preprocessing layer still needs richer normalization and split-generation logic
- raw episodes now include effective per-episode sample rates, and the new preprocessing path is designed to use those values instead of assuming the old fixed camera regime
- the collector is more recoverable now, but the underlying Myo device/serial transport can still fault after several episodes on some machines
- the GUI is now isolated from hardware faults by a worker process, but backend recovery behavior is still an area to keep tightening

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
- palm-frame rotation normalization using the wrist, index MCP, and pinky MCP
- train-split EMG standardization
- train-split target standardization
- configurable target representations:
  - direct point regression
  - per-finger articulation-angle regression

This is intentionally cleaner than the older project, where most preprocessing behavior was effectively collapsed into `dataset.py` and `utils.py`. The new repo is separating raw discovery, loading, transforms, and preprocessing assembly so later training and runtime inference can reuse the same interfaces cleanly.

### Current Representation Direction
The current preferred configuration for the Leap-based dataset is:
1. `target_mode=xyz`
2. palm-frame pose normalization enabled
3. EMG standardization enabled
4. direct point regression as the first baseline

The newer joint-angle target mode is now structurally supported and should be treated as the next comparison mode rather than the immediate default.

### Recommended Implementation Order
1. finish tightening collection and dataset contracts
2. finish the preprocessing layer:
   - episode validation reporting
   - split generation
   - sliding-window dataset objects
   - additional normalization controls
3. add training and evaluation modules with configurable `xyz` or `xy` targets
4. rebuild streaming inference and downstream control on top of the shared modules
