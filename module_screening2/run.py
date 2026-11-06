"""Dispatch one physical copy in a separate Python interpreter."""
from pathlib import Path
import argparse
import os
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from module_screening2.protocol import FOLDERS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=FOLDERS)
    args = parser.parse_args(sys.argv[1:2])
    entry = ROOT / FOLDERS[args.experiment] / 'PUFM-main/screening.py'
    command = [sys.executable, '-B', str(entry), *sys.argv[2:]]
    if os.name != 'nt':
        # Same PID on the server: queue SIGTERM reaches the actual training job.
        os.execv(sys.executable, command)
    child = subprocess.Popen(command)
    signal.signal(signal.SIGTERM, lambda *_: child.terminate())
    try:
        raise SystemExit(child.wait())
    except KeyboardInterrupt:
        child.terminate()
        child.wait()
        raise SystemExit(130)
