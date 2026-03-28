# Scripts Package

This folder contains thin package entry points for user-facing commands.

## Files

- `__init__.py`: Marks the scripts package.
- `collection_gui.py`: Launches the desktop collection GUI.
- `leap_myo_data_collection.py`: Launches the terminal collector fallback.
- `visualize_leap.py`: Launches the Leap visualizer.
- `visualize_myo.py`: Launches the Myo visualizer.
- `preprocess_dataset.py`: Small preprocessing entry point for manifest and dataset-pipeline checks.
- `replay_dataset.py`: Launches the dataset reviewer GUI for replaying and curating saved episodes.
- `train.py`: Hydra-backed training entry point for packaged model training.
- `evaluate.py`: Hydra-backed checkpoint evaluation entry point for packaged models.
