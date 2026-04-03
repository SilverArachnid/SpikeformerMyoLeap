#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Phase 2 coarse search for Transformer.
# Important exposed knobs:
# - num_layers
# - heads
# - embed_dim
# - window_size

run_experiment transformer l2_h2_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_layers=2 \
    model.model_kwargs.heads=2 \
    model.model_kwargs.embed_dim=64

run_experiment transformer l4_h4_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_layers=4 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=64

run_experiment transformer l6_h4_e64_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_layers=6 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=64

run_experiment transformer l4_h2_e96_w64 \
    dataset=default \
    dataset.window_size=64 \
    model.model_kwargs.num_layers=4 \
    model.model_kwargs.heads=2 \
    model.model_kwargs.embed_dim=96

run_experiment transformer l4_h4_e96_w96 \
    dataset=default \
    dataset.window_size=96 \
    dataset.preprocessing.emg_window_size=96 \
    model.model_kwargs.num_layers=4 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=96

run_experiment transformer l4_h4_e64_w128 \
    dataset=default \
    dataset.window_size=128 \
    dataset.preprocessing.emg_window_size=128 \
    model.model_kwargs.num_layers=4 \
    model.model_kwargs.heads=4 \
    model.model_kwargs.embed_dim=64
