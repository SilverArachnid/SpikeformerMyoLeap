# Training Package

This folder contains the training-facing dataset adapters, config objects, and training/evaluation loops.

## Files

- `__init__.py`: Re-exports the public training interfaces.
- `config.py`: Dataclasses for dataset, split, training, and evaluation settings.
- `datasets.py`: Windowed dataset builders on top of the preprocessing layer.
- `datasets.py`: Windowed dataset builders on top of the preprocessing layer, including train-split normalization fitting.
- `train.py`: Reusable training loop entry point.
- `evaluate.py`: Reusable checkpoint evaluation entry point.

## Notes

- `DatasetConfig.include_paths` can be used to restrict training or evaluation to a selected subset of the dataset.
- Each include path may point to:
  - a higher-level subtree such as `user_1/session_2/test_pose`
  - or an individual `ep_XXXX` directory
- When `include_paths` is empty, the full `dataset_root` is scanned for complete episodes.
- Train/validation splitting is now done at the episode level before window extraction to avoid leakage between adjacent windows from the same recording.
- EMG normalization and target standardization are now fit on the train split and reused consistently for validation and checkpoint evaluation.
- Preprocessing now supports:
  - point targets (`target_representation=points`)
  - joint-angle targets (`target_representation=joint_angles`)
  - palm-frame pose normalization for rotation-invariant 3D training
- Hydra YAML configs for training and evaluation live under `src/spikeformer_myo_leap/training/conf/`.
- Model, dataset, and split presets can be swapped with Hydra overrides such as `model=transformer` or `dataset=user1_session2_test_pose`.
- Training and evaluation now print progress bars, epoch summaries, and runtime information so long-running jobs are visible from the terminal.
- Training also supports full-episode validation on held-out episodes, with optional GIF visualizations saved every `N` epochs.
- Checkpoints now store normalization statistics so evaluation can apply the same feature scaling as training.
- Useful Hydra presets now include:
  - `dataset=default` for 3D point regression
  - `dataset=default_joint_angles` for 3D angle regression
  - `dataset=default_xy_compat` for old-style 2D compatibility runs
