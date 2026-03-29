# Training Config Package

This folder contains Hydra YAML configs for packaged training and evaluation.

## Files and folders

- `train.yaml`: Top-level training config.
- `evaluate.yaml`: Top-level evaluation config.
- `model/`: Model-family presets such as Spikeformer, Transformer, CNN-LSTM, CNN, and spiking CNN.
- `dataset/`: Dataset-selection presets, including:
  - default 3D point regression
  - 3D joint-angle regression
  - `xy` compatibility mode
  - selected-subtree examples
- `split/`: Train/validation split presets.

## Current dataset presets

- `dataset=default`: full dataset, `xyz` point targets, palm-frame normalization on
- `dataset=default_joint_angles`: full dataset, `xyz` joint-angle targets
- `dataset=default_xy_compat`: full dataset, `xy` point targets for older compatibility checks
- `dataset=user1_session2_test_pose`: example selected-subtree dataset using the same default 3D point preprocessing
