# CBS 2026 Experiments

This folder contains the tracked experiment scaffolding for the CBS 2026 paper.

## Goals

- keep ablations reproducible
- avoid losing results when training crashes
- keep model comparisons fair by documenting exact settings
- persist logs and a run registry outside ad hoc terminal history

## Structure

- `common.sh`: shared runner utilities for all paper experiments
- `phase1_representation_transformer.sh`: target-representation selection with Transformer
- `phase2_architecture_transformer.sh`: coarse Transformer architecture sweep
- `phase2_architecture_spikeformer.sh`: coarse Spikeformer architecture sweep
- `phase2_architecture_cnn_lstm.sh`: coarse CNN-LSTM architecture sweep
- `phase2_architecture_cnn.sh`: coarse CNN architecture sweep
- `phase2_architecture_spiking_cnn.sh`: coarse Spiking CNN sweep
- `logs/`: per-run terminal logs, created automatically
- `runs/`: base output directories passed to `train.py`, created automatically
- `results_registry.jsonl`: append-only run registry, created automatically

## Logging contract

Each scripted run records:

- timestamped run id
- model family
- human-readable label
- exact train command
- git commit hash
- exit code and final status
- log path
- resolved training artifact directory when available

The registry is append-only JSONL so interrupted runs are still visible.

## Recommended usage order

1. Start with `phase1_representation_transformer.sh`.
2. Compare:
   - `points_xyz` wrist-relative only
   - `points_xyz` palm-frame normalized
   - `joint_angles`
3. Freeze the preferred representation.
4. Run the phase-2 model family scripts.
5. Select top configurations per family for later exploit/final comparison runs.

## Notes

- The scripts intentionally favor clarity over automation complexity.
- They are designed for long unattended runs, not interactive tuning.
- If a model crashes, the log and registry entry should still survive.
- The search space here is limited to hyperparameters currently exposed cleanly by the model implementations.
- If later paper analysis requires deeper width/kernel ablations, expose those knobs in model configs first rather than overloading these scripts with hidden assumptions.
