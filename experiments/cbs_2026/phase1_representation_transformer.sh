#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Phase 1:
# choose the most promising target representation using one strong baseline.

run_experiment transformer points_wrist_w64 \
    dataset=default \
    dataset.preprocessing.use_palm_frame_pose=false \
    dataset.window_size=64

run_experiment transformer points_palm_w64 \
    dataset=default \
    dataset.preprocessing.use_palm_frame_pose=true \
    dataset.window_size=64

run_experiment transformer joint_angles_w64 \
    dataset=default_joint_angles \
    dataset.window_size=64
