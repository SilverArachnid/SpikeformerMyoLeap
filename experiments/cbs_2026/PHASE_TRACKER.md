# CBS 2026 Phase Tracker

This file records the outcome of each completed experiment phase and the exact
run folders that belong to that phase.

Use this as the human-readable companion to:

- `results_registry.jsonl`
- `logs/`
- `runs/`

## Phase 1: Representation Sweep

Script:

- `phase1_representation_transformer.sh`

Goal:

- choose the most promising target representation before running broader model-family sweeps

Completed runs:

1. Wrist-relative 3D points
   - label: `points_wrist_w64`
   - run folder: `runs/transformer/20260403_170611_transformer_xyz_points`
   - notes: earlier interrupted rerun exists in the registry, but this is the successful run to keep

2. Palm-frame 3D points
   - label: `points_palm_w64`
   - run folder: `runs/transformer/20260403_171951_transformer_xyz_points`

3. Joint angles
   - label: `joint_angles_w64`
   - run folder: `runs/transformer/20260403_173351_transformer_xyz_joint_angles`

Summary:

- palm-frame normalization clearly improved the coordinate-based representation
  over wrist-relative coordinates alone
- joint-angle training was stable and remains a valid secondary representation
  track

Observed metrics:

### Wrist-relative points

- `best_val_loss = 0.3775`
- full-episode `rmse = 4.9601`
- full-episode `mae = 3.4130`
- full-episode `fps = 1645.2`

### Palm-frame points

- `best_val_loss = 0.2632`
- full-episode `rmse = 4.1227`
- full-episode `mae = 2.3522`
- full-episode `fps = 1581.4`

### Joint angles

- `best_val_loss = 0.1483`
- full-episode `rmse = 0.0729`
- full-episode `mae = 0.0442`
- full-episode `fps = 1630.8`

Interpretation:

- use palm-frame normalized `points_xyz` as the default point-space
  representation for phase 2
- keep `joint_angles` as a secondary track, but do not directly compare its loss
  or RMSE/MAE values against point-space metrics because the target space and
  units are different

Decision for next phase:

- phase 2 main track: `points_xyz + palm-frame`
- phase 2 secondary track: `joint_angles`

## Phase 2: Architecture Sweep

Status:

- not started

Planned scripts:

- `phase2_architecture_transformer.sh`
- `phase2_architecture_spikeformer.sh`
- `phase2_architecture_cnn_lstm.sh`
- `phase2_architecture_cnn.sh`
- optional: `phase2_architecture_spiking_cnn.sh`

Expected outcome:

- shortlist top configurations per model family
- compare architecture accuracy-speed tradeoffs under the chosen
  representation
