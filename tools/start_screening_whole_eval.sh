#!/usr/bin/env bash
set -euo pipefail

TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_REPO"
export V3_DATA_ROOT="${V3_DATA_ROOT:-$TASK_REPO/adaptive_resampling_data_v3}"
export SCREENING_HISTORY_ROOT="${SCREENING_HISTORY_ROOT:-$TASK_REPO/server_runs}"
export SCREENING_RUNS_ROOT="${SCREENING_RUNS_ROOT:-$SCREENING_HISTORY_ROOT/patch32}"
export MODELNET40_WHOLE_DATA_ROOT="${MODELNET40_WHOLE_DATA_ROOT:-$TASK_REPO/modelnet40_object80_seed21}"
TASK_OUTPUT="${SCREENING_WHOLE_OUTPUT:-$SCREENING_HISTORY_ROOT/whole_module_eval80_20261110_v2}"
TASK_LOG="${TASK_OUTPUT}.console.log"
mkdir -p -- "$(dirname -- "$TASK_OUTPUT")"

nohup python -u -B tools/evaluate_screening_whole.py --output "$TASK_OUTPUT" "$@" \
  >> "$TASK_LOG" 2>&1 < /dev/null &
TASK_PID=$!
printf 'Queue PID: %s\nLog: %s\nResults: %s\n' "$TASK_PID" "$TASK_LOG" "$TASK_OUTPUT"
printf 'Follow progress: tail -f "%s"\n' "$TASK_LOG"
