#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ADAPTIVE_PYTHON:-python}" -u -B "$script_dir/adaptive_v3.py" "$@"
