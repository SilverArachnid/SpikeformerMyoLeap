# Training Package

This folder contains the training-facing dataset adapters, config objects, and training/evaluation loops.

## Files

- `__init__.py`: Re-exports the public training interfaces.
- `config.py`: Dataclasses for dataset, split, training, and evaluation settings.
- `datasets.py`: Windowed dataset builders on top of the preprocessing layer.
- `train.py`: Reusable training loop entry point.
- `evaluate.py`: Reusable checkpoint evaluation entry point.

## Notes

- `DatasetConfig.include_paths` can be used to restrict training or evaluation to a selected subset of the dataset.
- Each include path may point to:
  - a higher-level subtree such as `user_1/session_2/test_pose`
  - or an individual `ep_XXXX` directory
- When `include_paths` is empty, the full `dataset_root` is scanned for complete episodes.
