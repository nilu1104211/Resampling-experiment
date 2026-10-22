#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_v3.sh {pufmpp|spu-imr|ounet} COMMAND [OPTIONS]" >&2
    exit 2
fi
case "$1" in
    pufmpp) folder=Enhanced_PUFM-main ;;
    spu-imr) folder=spu-imr-main ;;
    ounet) folder=upsample-clean-master ;;
    *) echo "Unknown model: $1 (expected pufmpp, spu-imr or ounet)" >&2; exit 2 ;;
esac
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${V3_PYTHON:-python}" -u -B "$script_dir/$folder/v3.py" "$@"
