# CLAUDE.md — SpikeformerMyoLeap

## Project Summary

EMG-to-hand-pose regression pipeline. Collects synchronized data from a **Myo armband** (8-channel EMG, 200 Hz) and **Leap Motion Controller** (~100–120 Hz, 21-joint 3D pose), trains deep learning models to predict hand pose from EMG alone, and (planned) runs streaming inference for real-time gesture control.

---

## Environment Setup

```bash
# First time — installs Ultraleap system deps, creates venv, builds Leap bindings
bash setup_env.sh

# Every session
source .venv/bin/activate
```

**Runtime env vars required for Leap:**
```bash
export LEAPSDK_INSTALL_LOCATION="/usr/lib/ultraleap-hand-tracking-service"
export LEAPC_HEADER_OVERRIDE="/usr/include/LeapC.h"
export LEAPC_LIB_OVERRIDE="/usr/lib/x86_64-linux-gnu/libLeapC.so"
```

**Package manager:** `uv` (lockfile at `uv.lock`)

```bash
uv pip install -e .        # install/reinstall package in editable mode
uv pip install <pkg>       # add a dependency
```

---

## All Commands

### Data Collection

```bash
# GUI collector (primary) — PySide6 app with live previews
python src/spikeformer_myo_leap/scripts/collection_gui.py
# backward-compat wrapper:
python collection_gui.py

# Terminal collector (fallback, no GUI)
python src/spikeformer_myo_leap/scripts/leap_myo_data_collection.py
# backward-compat wrapper:
python leap_myo_data_collection.py
```

### Visualization / Hardware Preview

```bash
# Live Leap pose viewer
python src/spikeformer_myo_leap/scripts/visualize_leap.py
python visualize_leap.py  # wrapper

# Live Myo EMG viewer
python src/spikeformer_myo_leap/scripts/visualize_myo.py
python visualize_myo.py   # wrapper

# Local dark-mode dashboard (recorded episode review)
python local_visualizer.py
```

### Preprocessing

```bash
# Validate and report on dataset episodes
python src/spikeformer_myo_leap/scripts/preprocess_dataset.py
python preprocess_dataset.py  # wrapper
```

### Training

```bash
# Train with defaults (Spikeformer, default dataset, 50 epochs)
python src/spikeformer_myo_leap/scripts/train.py
python train.py  # wrapper

# Override model
python train.py model=transformer
python train.py model=cnn_lstm
python train.py model=cnn
python train.py model=spiking_cnn
python train.py model=spikeformer

# Override hyperparameters (Hydra syntax)
python train.py num_epochs=100 learning_rate=0.0005 batch_size=32

# Override dataset
python train.py dataset=user1_session2_test_pose

# Override model kwargs inline
python train.py "model.model_kwargs.embed_dim=128" "model.model_kwargs.heads=8"

# Change output dir
python train.py output_dir=artifacts/my_run

# Full example: Spikeformer sweep
python train.py model=spikeformer "model.model_kwargs.heads=2" "model.model_kwargs.num_blocks=3"
python train.py model=spikeformer "model.model_kwargs.heads=4" "model.model_kwargs.num_blocks=4"
python train.py model=spikeformer "model.model_kwargs.heads=4" "model.model_kwargs.num_blocks=6"
```

Training outputs land in `artifacts/train/` (or `outputs/` for older runs).
Checkpoints: `<output_dir>/<run_id>/checkpoint.pt`
Full-episode GIF evals run every 5 epochs by default.

### Evaluation

```bash
# Evaluate a saved checkpoint
python src/spikeformer_myo_leap/scripts/evaluate.py checkpoint_path=artifacts/train/<run>/checkpoint.pt
python evaluate.py checkpoint_path=...  # wrapper

# Override model config for eval
python evaluate.py model=transformer checkpoint_path=...
```

---

## Hydra Config Structure

```
src/spikeformer_myo_leap/training/conf/
├── train.yaml          # top-level train defaults
├── evaluate.yaml       # top-level evaluate defaults
├── model/
│   ├── spikeformer.yaml
│   ├── transformer.yaml
│   ├── cnn_lstm.yaml
│   ├── cnn.yaml
│   └── spiking_cnn.yaml
├── dataset/
│   ├── default.yaml
│   └── user1_session2_test_pose.yaml
└── split/
    └── default.yaml    # train_fraction: 0.8, seed: 42
```

**Default training config:**
- `num_epochs: 50`, `batch_size: 64`, `learning_rate: 0.0001`
- `window_size: 64`, `stride: 1`
- `target_mode: xyz` (21 joints × 3 = 63 outputs)
- `resample_hz: 100.0`
- `use_wrist_relative_pose: true`
- `full_episode_eval: every 5 epochs`

**Default Spikeformer config:** `embed_dim=64`, `heads=4`, `num_blocks=4`, `mlp_ratio=2.0`

---

## Package Layout

```
src/spikeformer_myo_leap/
├── app/
│   └── collection_gui.py     # PySide6 GUI
├── collection/
│   ├── controller.py         # collection orchestration + auto-recovery
│   ├── worker.py             # threaded Leap/Myo I/O
│   └── terminal.py           # terminal fallback collector
├── config/
│   └── preprocessing.py      # preprocessing config dataclass
├── data/
│   ├── contracts.py          # shared constants, landmark names, EMG channel count
│   ├── io.py                 # episode save-path helpers, episode writer
│   ├── raw.py                # raw episode discovery
│   ├── manifest.py           # manifest building across episodes
│   ├── loaders.py            # CSV → NumPy array loaders
│   ├── transforms.py         # wrist-relative normalization
│   └── preprocessing.py      # resample, sync, preprocessing pipeline
├── models/
│   ├── registry.py           # name → class mapping
│   ├── spikeformer.py        # corrected multi-head spike attention
│   ├── transformer.py        # standard Transformer baseline
│   ├── cnn_lstm.py           # CNN-LSTM regressor
│   ├── cnn.py                # temporal CNN baseline
│   └── spiking_cnn.py        # spiking CNN baseline
├── training/
│   ├── config.py             # training/eval config dataclasses
│   ├── datasets.py           # windowed dataset + episode-level splits
│   ├── train.py              # training loop, checkpointing
│   ├── evaluate.py           # checkpoint evaluation
│   ├── full_episode.py       # full-episode validation + GIF output
│   └── conf/                 # Hydra YAML configs (see above)
├── visualization/
│   ├── local_dashboard.py    # Matplotlib dark-mode dashboard (primary)
│   ├── rerun.py              # Rerun viewer (optional, Linux GPU issues)
│   ├── leap_viewer.py        # live Leap preview
│   └── myo_viewer.py         # live Myo preview
└── scripts/                  # package entry points (import these, not top-level wrappers)
```

**Top-level `.py` wrappers** (`train.py`, `evaluate.py`, `collection_gui.py`, etc.) are backward-compat shims — prefer the `scripts/` versions.

---

## Data Layout

```
datasets/<subject_id>/<session_name>/<pose_name>/ep_NNNN/
    emg.csv       # 8 channels, timestamped
    pose.csv      # 21 joints × XYZ, timestamped
    meta.json     # effective_emg_hz, effective_pose_hz, duration, ...
```

**Current data:** `datasets/Pranav/session_3/hand_random/` — 19 episodes

---

## Architecture Notes

**Spikeformer was fixed in V2:**
- V1 stored `heads` but never split the embedding → effectively single-head
- V2 uses explicit head splitting; `heads` now meaningfully changes behavior
- Old hyperparameter sweeps over `heads` are unreliable

**Spiking time-axis was fixed:**
- V1 fed `[B, C, T]` directly to `MultiStepLIFNode` (wrong axis)
- V2 explicitly permutes so temporal axis aligns with spiking neuron expectations

**Visualization backend:**
- Use **local Matplotlib dashboard** by default
- Rerun is available but has Linux GPU stability issues

**Target mode:**
- `xyz` (default): 21 joints × 3 = 63-dimensional output
- `xy` (compatibility only): 21 joints × 2 = 42-dimensional output

---

## Implementation Status

| Area | Status |
|------|--------|
| Hardware collection + GUI | Done |
| Collector auto-recovery | Done |
| Raw data pipeline (manifest, loaders) | Done |
| Preprocessing (resample, normalize, window) | Done |
| All 5 model architectures | Done |
| Training loop + Hydra configs + checkpointing | Done |
| Full-episode eval + GIF visualization | Done |
| Preprocessing validation reporting | Partial — episodes silently skipped |
| Streaming inference runtime | Not started — `feat/streaming-inference-runtime` branch scaffolded |
| Robot/simulation integration | Future phase |

**Next up (priority order):**
1. Preprocessing validation/reporting (explicit failure output)
2. Streaming inference runtime (`src/spikeformer_myo_leap/inference/`)
3. Training refinements (split reporting, more normalization options)

---

## Recommended First Baseline Run

```bash
python train.py \
  model=spikeformer \
  "model.model_kwargs.embed_dim=64" \
  "model.model_kwargs.heads=4" \
  "model.model_kwargs.num_blocks=4" \
  num_epochs=50
```

**Comparison grid:**
```bash
# Small
python train.py model=spikeformer "model.model_kwargs.heads=2" "model.model_kwargs.num_blocks=3"
# Baseline
python train.py model=spikeformer "model.model_kwargs.heads=4" "model.model_kwargs.num_blocks=4"
# Deeper
python train.py model=spikeformer "model.model_kwargs.heads=4" "model.model_kwargs.num_blocks=6"
```

---

## Planned Inference Package Layout

*(Not yet implemented — from ROADMAP_ISSUES.md)*

```
src/spikeformer_myo_leap/inference/
    runtime.py
    buffers.py
    predictor.py
    model_loader.py
    model_adapters.py
    visualization.py
scripts/run_inference.py
```
