#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Phase 2 coarse search for Spikeformer.
# Important exposed knobs:
# - num_blocks
# - heads
# - embed_dim
# - window_size

run_experiment spikeformer b2_h2_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=2 \
    model.model_kwargs.heads=2 \
    model.model_kwargs.embed_dim=64

run_experiment spikeformer b4_h4_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=4 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=64

run_experiment spikeformer b6_h4_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=6 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=64

run_experiment spikeformer b4_h2_e96_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_blocks=4 \
    model.model_kwargs.heads=2 \
    model.model_kwargs.embed_dim=96

run_experiment spikeformer b4_h4_e96_w96 \
    dataset=default \
    dataset.window_size=96 \
    dataset.preprocessing.emg_window_size=96 \
    model.model_kwargs.num_blocks=4 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=96

run_experiment spikeformer b4_h4_e64_w128 \
    dataset=default \
    dataset.window_size=128 \
    dataset.preprocessing.emg_window_size=128 \
    model.model_kwargs.num_blocks=4 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=64
