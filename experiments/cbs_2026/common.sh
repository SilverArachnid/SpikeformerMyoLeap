#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/cbs_2026"
LOG_DIR="$EXPERIMENT_DIR/logs"
RUNS_DIR="$EXPERIMENT_DIR/runs"
REGISTRY_PATH="$EXPERIMENT_DIR/results_registry.jsonl"

mkdir -p "$LOG_DIR" "$RUNS_DIR"

run_experiment() {
    local family="$1"
    local label="$2"
    shift 2

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local run_id="${timestamp}_${family}_${label}"
    local family_run_dir="$RUNS_DIR/$family"
    local log_path="$LOG_DIR/${run_id}.log"
    local git_commit
    git_commit="$(git -C "$ROOT_DIR" rev-parse HEAD)"

    mkdir -p "$family_run_dir"

    local -a cmd=(
        uv run train.py
        "model=${family}"
        "output_dir=${family_run_dir}"
        "$@"
    )

    {
        echo "run_id=${run_id}"
        echo "family=${family}"
        echo "label=${label}"
        echo "git_commit=${git_commit}"
        printf 'command='
        printf '%q ' "${cmd[@]}"
        echo
        echo
    } >> "$log_path"

    local exit_code=0
    if (cd "$ROOT_DIR" && "${cmd[@]}") 2>&1 | tee -a "$log_path"; then
        exit_code=0
    else
        exit_code=$?
    fi

    local status="success"
    if [[ "$exit_code" -ne 0 ]]; then
        status="failed"
    fi

    local run_output_dir=""
    run_output_dir="$(grep -oP 'Saving run artifacts under \K.*' "$log_path" | tail -n 1 || true)"

    REGISTRY_PATH="$REGISTRY_PATH" \
    RUN_ID="$run_id" \
    FAMILY="$family" \
    LABEL="$label" \
    GIT_COMMIT="$git_commit" \
    STATUS="$status" \
    EXIT_CODE="$exit_code" \
    LOG_PATH="$log_path" \
    RUN_OUTPUT_DIR="$run_output_dir" \
    COMMAND="$(printf '%q ' "${cmd[@]}")" \
    python3 - <<'PY'
import json
import os
from datetime import datetime, timezone

record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ["RUN_ID"],
    "family": os.environ["FAMILY"],
    "label": os.environ["LABEL"],
    "git_commit": os.environ["GIT_COMMIT"],
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["EXIT_CODE"]),
    "log_path": os.environ["LOG_PATH"],
    "run_output_dir": os.environ["RUN_OUTPUT_DIR"],
    "command": os.environ["COMMAND"].strip(),
}
with open(os.environ["REGISTRY_PATH"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
PY

    return "$exit_code"
}
