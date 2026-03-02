# SpikeformerMyoLeap

Version 2 of the SpikeformerMyo project. This repository contains a standalone, robust pipeline for collecting synchronized 21-joint 3D hand poses via Leap Motion (`leapc-python-bindings`) and 8-channel EMG data from the Myo armband (`pyomyo`). 

## Features
- **Standalone Environment**: Uses `uv` and `pyproject.toml` to manage dependencies seamlessly, including local CFFI compilation for the Leap SDK.
- **Hydra Configuration**: Easily adjust recording durations, subjects, and parameters via `conf/config.yaml`.
- **Rerun SDK Visualizations**: Beautiful, real-time 3D skeletons and EMG timeseries plotting.

## Quickstart

1. Clone this repository adjacent to the `leapc-python-bindings` repository.
2. Setup the environment:
```bash
./setup_env.sh
source .venv/bin/activate
```
3. (Optional) Run standalone visualizers to test hardware:
```bash
uv run visualize_leap.py
uv run visualize_myo.py
```
4. Start data collection:
```bash
uv run leap_myo_data_collection.py
```
*Press Spacebar to record an episode, and ESC to quit.* By default, if `visualize: true` is set in the config, a Rerun stream will automatically attach to visualize the collection.
