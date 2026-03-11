# Data Package

This folder owns the dataset contract, raw episode IO, dataset discovery, and preprocessing scaffolding.

## Files

- `__init__.py`: Re-exports the primary data-layer interfaces.
- `contracts.py`: Shared constants and collection-oriented dataclasses such as landmark names and collection settings.
- `io.py`: Episode save-path helpers and episode writing utilities.
- `raw.py`: Raw episode discovery and raw CSV/metadata loading.
- `manifest.py`: Builds a structured manifest across saved episodes.
- `loaders.py`: Converts raw episode CSVs into NumPy arrays for EMG and pose targets.
- `transforms.py`: Shared pose-space transforms, such as wrist-relative conversion.
- `preprocessing.py`: Resampling and preprocessing pipeline entry points for training-ready episode tensors.
