#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Phase 2 coarse search for CNN-LSTM.
# Currently exposed important knobs:
# - hidden_dim
# - num_layers
# - window_size
#
# Conv widths are fixed in the current implementation and should only be added
# later if the family becomes central to the paper.

run_experiment cnn_lstm h64_l1_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.hidden_dim=64 \
    model.model_kwargs.num_layers=1

run_experiment cnn_lstm h128_l2_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.hidden_dim=128 \
    model.model_kwargs.num_layers=2

run_experiment cnn_lstm h128_l1_w96 \
    dataset=default \
    dataset.window_size=96 \
    dataset.preprocessing.emg_window_size=96 \
    model.model_kwargs.hidden_dim=128 \
    model.model_kwargs.num_layers=1

run_experiment cnn_lstm h64_l2_w128 \
    dataset=default \
    dataset.window_size=128 \
    dataset.preprocessing.emg_window_size=128 \
    model.model_kwargs.hidden_dim=64 \
    model.model_kwargs.num_layers=2
