#!/usr/bin/env bash
# Run inside the existing scdata-v3 environment. No dependency installs/changes.
set -euo pipefail

TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_REPO"
export V3_DATA_ROOT="${V3_DATA_ROOT:-$TASK_REPO/adaptive_resampling_data_v3}"
export PUFM_BASELINE_CHECKPOINT="${PUFM_BASELINE_CHECKPOINT:-$TASK_REPO/server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt}"
TASK_OUTPUT="${MODULE_SCREENING2_OUTPUT:-$TASK_REPO/server_runs/patch32/module_screening2/sequential15_v1}"
TASK_PYTHON="${MODULE_SCREENING2_PYTHON:-python}"
mkdir -p -- "$(dirname -- "$TASK_OUTPUT")"
TASK_LOG="${TASK_OUTPUT}.console.log"

nohup "$TASK_PYTHON" -u -B module_screening2/train_all.py \
  --epochs 15 --batch-size 8 --workers 0 --assignment-workers 8 \
  --fps-backend cuda --fused-optimizer --output "$TASK_OUTPUT" "$@" \
  >> "$TASK_LOG" 2>&1 < /dev/null &
TASK_PID=$!
printf 'Queue PID: %s\nLog: %s\nResults: %s\n' "$TASK_PID" "$TASK_LOG" "$TASK_OUTPUT"
printf 'Inspect progress: tail -f "%s"\n' "$TASK_LOG"
