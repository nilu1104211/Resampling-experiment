"""Dispatch each copy in a separate interpreter, with no shared `models` imports."""
from pathlib import Path
import argparse
import subprocess
import sys

FOLDERS = {
    'm1': 'm1_endpoint_cd', 'm2': 'm2_apml', 'm3': 'm3_gaussian_features',
    'm4': 'm4_repkpoints', 'm5': 'm5_adaptive_denoise',
    'm6': 'm6_grad_refine', 'm7': 'm7_ats',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=FOLDERS)
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    entry = Path(__file__).resolve().parent / FOLDERS[args.experiment] / 'PUFM-main/screening.py'
    raise SystemExit(subprocess.call([sys.executable, '-B', str(entry), *rest]))
