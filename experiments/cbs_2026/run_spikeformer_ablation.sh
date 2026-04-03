#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

run_experiment spikeformer points_palm_w64_b2 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=2

run_experiment spikeformer points_palm_w64_b4 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=4

run_experiment spikeformer joint_angles_w64_b4 \
    dataset=default_joint_angles \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=4

run_experiment spikeformer points_palm_w96_b4 \
    dataset=default \
    dataset.window_size=96 \
    dataset.preprocessing.emg_window_size=96 \
    model.model_kwargs.num_blocks=4
