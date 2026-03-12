# Research Notes

This document tracks architecture discussions, implementation findings, and experiment ideas that should inform future model and preprocessing work.

## Spikeformer Attention Correction

### Finding
In the older `/home/pranav/pranav/SpikeFormerMyo` implementation, the Spikeformer attention block stored a `heads` parameter but did not actually split the embedding into multiple heads during attention computation.

### Impact
- changing `heads` in the old implementation likely had little or no real effect
- the model still learned useful correlations, but it was not functioning as a true multi-head attention model
- old hyperparameter tuning around `heads` should be treated cautiously

### Current Direction
The new packaged Spikeformer implementation uses explicit head splitting so `heads` now changes the model behavior meaningfully.

## Spiking Time-Axis Handling

### Finding
The older spiking conv-style paths likely fed `[B, C, T]` tensors directly into `MultiStepLIFNode`, which expects a time-major layout.

### Impact
- temporal spiking dynamics may not have been applied over the real EMG time axis
- this can lead to models that appear to work somewhat but underperform and behave inconsistently

### Current Direction
The new spiking implementations explicitly permute tensors so the temporal axis is treated as time when passed to spiking neuron layers.

## Heads Recommendation

### Current Recommendation
- default: `4` heads
- smaller comparison: `2` heads
- minimal ablation: `1` head
- higher-capacity optional comparison: `8` heads only after the baseline is stable

### Reasoning
- the EMG-to-pose mapping likely benefits from multiple feature subspaces
- `4` heads is a good balance for `embed_dim=64`
- too many heads may over-fragment a relatively small embedding and dataset

## Spikeformer Depth Recommendation

### Current Recommendation
- default: `4` blocks
- smaller comparison: `3` blocks
- larger comparison: `6` blocks once the baseline is stable

### Reasoning
- `4` blocks is a balanced starting point for capacity and stability
- deeper models should be tried only after verifying that the corrected implementation trains cleanly

## Initial Spikeformer Baseline

Recommended first serious baseline:

- `embed_dim = 64`
- `heads = 4`
- `num_blocks = 4`
- target mode: `xyz`
- preprocessing resample rate: `100 Hz`
- EMG window size: `64`

## Comparison Grid To Try

Suggested early comparisons:

1. `heads=2`, `num_blocks=3`
2. `heads=4`, `num_blocks=4`
3. `heads=4`, `num_blocks=6`

Optional later comparison:

4. `heads=8`, `num_blocks=4`

## Preprocessing Notes

### Old Pipeline
- older camera pose data was sparse, around `~30 fps`
- pose had to be strongly upsampled to match the training timebase

### New Pipeline
- Leap pose is already much denser, roughly `~100–120 Hz`
- resampling to `100 Hz` is mild alignment rather than aggressive interpolation
- this makes `xyz` targets much more defensible

### Current Recommendation
- keep `100 Hz` as the initial training timebase
- keep window size at `64`
- use `xyz` as the default target mode
- keep `xy` as a compatibility option only

## Questions To Revisit

- whether wrist-relative Cartesian `xyz` is the best final target representation
- whether joint-angle or kinematic targets outperform raw Cartesian targets
- whether `200 Hz` plus a larger window is better once the dataset is larger
- whether the corrected Spikeformer materially outperforms the Transformer baseline
