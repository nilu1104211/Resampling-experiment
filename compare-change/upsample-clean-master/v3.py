"""V3 comparison entry point. See ../V3_ADAPTATION.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v3_common import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent, 'ounet')
