# CBS 2026 Plan

## Working direction

Old paper framing is too centered on spiking networks and too tied to the 2D MediaPipe pipeline. The new paper should be centered on:

- continuous EMG-to-3D hand-state regression
- representation choice as a first-class factor
- model comparison across ANN and SNN families
- live inference and prosthetic retargeting feasibility

The paper should not be sold as "Spikformer for prosthetic control" only. It should be sold as a broader study of realtime EMG-to-hand-state regression for prosthetic retargeting.

## Candidate title directions

Possible title directions:

- Continuous EMG-to-3D Hand-State Regression for Realtime Prosthetic Retargeting
- Realtime EMG-to-Hand-State Regression for Prosthetic Control: Representation, Architecture, and Retargeting
- Continuous EMG-to-3D Hand Regression for Prosthetic Hand Control with ANN and SNN Models

Avoid titles that imply:

- clinical validation
- amputee deployment
- neuromorphic hardware evaluation
- Spikformer-only novelty

## Proposed paper contributions

Target three contributions:

1. A 3D Myo + Leap continuous EMG-to-hand-state regression pipeline with online/offline preprocessing parity.
2. A representation study comparing normalized 3D point targets and articulation-oriented joint-angle targets.
3. A live inference and prosthetic retargeting demonstration with model-speed analysis.

## Main experimental question

Primary question:

- What representation and model choices best support accurate and realtime EMG-to-hand-state regression for prosthetic retargeting?

Secondary questions:

- Does palm-frame normalization improve point-based regression?
- Are joint-angle targets better than point targets for prosthetic-oriented control?
- Which architectures provide the best accuracy-speed tradeoff?
- Are transformer-based models helped by longer temporal context?

## Recommended scope

Keep the paper focused. Do not try to publish every feature in the repo.

Include:

- Myo + Leap dataset and preprocessing
- point vs angle targets
- ANN vs SNN model comparison
- live inference / retargeting demonstration

De-emphasize or move to background/future work:

- old 2D MediaPipe pipeline
- old gesture classification study
- neuromorphic deployment claims
- broad prosthetic universality claims

## How much of the old work to mention

### 2D approach

Mention briefly as historical motivation only:

- initial 2D MediaPipe approach exposed the weakness of depth-free retargeting
- current repo moves to Leap-based 3D hand state to address that limitation

Do not make the 2D approach a main result section unless it is needed for a historical comparison figure.

### Gesture classification

Mention very briefly, likely in one paragraph in methodology or background:

- gesture classification was an initial feasibility step
- the main work is continuous regression

Do not keep a large gesture-classification results section in the new paper unless the conference page limit allows it and it supports the story clearly.

## Model set to keep

Prioritize a smaller, defensible model set:

- Transformer
- CNN-LSTM
- Spikeformer
- CNN

Optional:

- Spiking CNN, only if it adds a useful contrast

If page budget is tight, focus on the top 3:

- Transformer
- CNN-LSTM
- Spikeformer

## Representation ablation matrix

This should be one of the main result tables.

Required target representations:

1. `points_xyz` + wrist-relative only
2. `points_xyz` + palm-frame normalization
3. `joint_angles`

Optional:

4. `points_xy_compat`

Recommendation:

- Run representation ablations first with Transformer
- Choose the best representation for the broader model comparison

## Architecture ablation matrix

After selecting a preferred representation, compare:

- Transformer
- CNN-LSTM
- CNN
- Spikeformer
- optional Spiking CNN

Metrics:

- validation loss
- RMSE
- MAE
- full-episode metrics
- pure inference FPS

## Hyperparameter exploration strategy

We should not compare models with obviously weak settings.

Use a staged exploration/exploitation approach:

### Stage 1: coarse exploration

Per model family, define a small coarse grid.

Transformer:

- window size: 64, 96, 128
- embed dim: 64, 96
- layers: 2, 4, 6
- heads: 2, 4
- ff mult: 2, 4

CNN-LSTM:

- hidden dim: 64, 128
- recurrent layers: 1, 2
- window size: 64, 96, 128

CNN:

- embed dim: 64, 96
- blocks: 3, 4, 5
- window size: 64, 96, 128

Spikeformer:

- embed dim: 64, 96
- blocks: 2, 4, 6
- heads: 2, 4
- window size: 64, 96, 128
- update rate for live deployment tracked separately

Spiking CNN:

- embed dim: 64, 96
- blocks: 3, 4, 5
- window size: 64, 96, 128

### Stage 2: exploit best region

For each family:

- keep top 2-3 settings from Stage 1
- rerun with more epochs and repeated seeds if feasible

### Stage 3: final fair comparison

Freeze one best config per model family and compare on the same split.

## Logging and reproducibility requirements

Need a robust ablation runner so training crashes do not lose information.

Required outputs for every run:

- exact config snapshot
- git commit hash
- start/end time
- train/val metrics by epoch
- full-episode metrics
- inference FPS
- path to checkpoints
- path to qualitative outputs
- status marker: success / failed / interrupted

Recommended implementation:

- per-model bash launcher scripts under a new experiment folder
- append-only CSV or JSONL experiment registry
- one run directory per training job
- redirect stdout/stderr to persistent log files

## Reading list to understand each model deeply

### Core papers

Transformer fundamentals:

- Attention Is All You Need
- a strong PyTorch transformer implementation reference

EMG + transformer / sequential control:

- TraHGR
- Vangi et al. CNN-LSTM regression paper
- emg2pose
- Leroux et al. Online Transformers with Spiking Neurons for Fast Prosthetic Hand Control

Spiking foundations:

- a good SNN review
- surrogate gradient basics
- LIF neuron formulations used by the library

Spikeformer-specific:

- Spikeformer: When Spiking Neural Network Meets Transformer
- Spikformer V2

Need to understand and document:

- what SPS is doing
- what spiking self-attention is actually replacing
- how spike generation happens in our implementation
- which parts were inherited from image classification and may not be necessary for 1D EMG regression

## Long-context push goal

This is a useful secondary experiment, but not the first priority.

Questions:

- do transformer-based models benefit from longer context?
- can longer windows improve difficult multi-finger sequences?
- does autoregressive context or past predicted actions help?

Recommended late-stage experiments:

- window size 64 vs 96 vs 128 vs 160
- compare Transformer and Spikeformer only
- optionally test feeding previous predicted articulation state

Only run this after the main representation/model comparison is stable.

## Figures to keep or remake

Likely needed:

- updated data collection setup with Myo + Leap
- updated preprocessing pipeline figure
- updated model overview figure
- one qualitative full-episode prediction figure
- one live inference / prosthetic retargeting figure

Do not reuse the old 2D mapping figure as a main figure unless explicitly labeled historical.

## Concrete execution order

1. Finalize paper framing and title direction.
2. Build experiment folder and robust ablation logging.
3. Lock one dataset version and split protocol.
4. Run representation ablations with Transformer.
5. Select best representation.
6. Run architecture comparison on that representation.
7. Run focused hyperparameter exploitation for the top 2-3 models.
8. Run live inference / retargeting evaluation on the best models.
9. Run optional long-context experiments.
10. Rebuild figures and write paper around the new results.
