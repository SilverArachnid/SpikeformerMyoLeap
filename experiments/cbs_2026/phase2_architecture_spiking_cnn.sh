#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Phase 2 coarse search for Spiking CNN.
# Currently exposed important knobs:
# - num_blocks
# - embed_dim
# - window_size
#
# Backend is intentionally fixed outside the main paper comparison.

run_experiment spiking_cnn b3_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=3 \
    model.model_kwargs.embed_dim=64

run_experiment spiking_cnn b4_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=4 \
    model.model_kwargs.embed_dim=64

run_experiment spiking_cnn b5_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=5 \
    model.model_kwargs.embed_dim=64

run_experiment spiking_cnn b4_e96_w96 \
    dataset=default \
    dataset.window_size=96 \
    dataset.preprocessing.emg_window_size=96 \
    model.model_kwargs.num_blocks=4 \
    model.model_kwargs.embed_dim=96
