#!/usr/bin/env bash
# Source this file from Bash after activating the server's Conda environment.
export V3_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export V3_DATA_ROOT="${V3_DATA_ROOT:-$V3_CODE_ROOT/adaptive_resampling_data_v3}"
v3() { bash "$V3_CODE_ROOT/compare-change/run_v3.sh" "$@"; }
ours() { bash "$V3_CODE_ROOT/our/PUFM-main/run_adaptive.sh" "$@"; }
