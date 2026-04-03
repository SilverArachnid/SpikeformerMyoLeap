#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

run_experiment cnn points_palm_w64 \
    dataset=default \
    dataset.window_size=64

run_experiment cnn points_palm_w96 \
    dataset=default \
    dataset.window_size=96 \
    dataset.preprocessing.emg_window_size=96

run_experiment cnn joint_angles_w64 \
    dataset=default_joint_angles \
    dataset.window_size=64
