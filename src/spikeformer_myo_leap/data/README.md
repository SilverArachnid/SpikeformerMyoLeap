# Data Package

This folder owns the dataset contract, raw episode IO, dataset discovery, and preprocessing scaffolding.

## Files

- `__init__.py`: Re-exports the primary data-layer interfaces.
- `contracts.py`: Shared constants and collection-oriented dataclasses such as landmark names and collection settings.
- `io.py`: Episode save-path helpers and episode writing utilities.
- `raw.py`: Raw episode discovery and raw CSV/metadata loading.
- `manifest.py`: Builds a structured manifest across saved episodes.
- `loaders.py`: Converts raw episode CSVs into NumPy arrays for EMG and pose targets.
- `transforms.py`: Shared pose-space transforms, palm-frame normalization, joint-angle conversion, and train/eval standardization helpers.
- `preprocessing.py`: Resampling and preprocessing pipeline entry points for training-ready episode tensors.

## Current preprocessing modes

- Coordinate modes:
  - `xyz`: full 3D joint coordinates
  - `xy`: compatibility-only 2D coordinates
- Target representations:
  - `points`: regress joint coordinates directly
  - `joint_angles`: regress per-finger articulation angles in radians

## Current normalization flow

- Wrist-relative translation can be applied first.
- Palm-frame rotation normalization can then remove global hand orientation using:
  - wrist
  - index MCP
  - pinky MCP
- EMG standardization is fit on the train split and then reused for validation/evaluation.
- Target standardization is also fit on the train split and applied consistently across train/val/eval.
