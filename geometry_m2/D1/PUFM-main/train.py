"""Independent D1 PUFM entry point."""
from pathlib import Path
import sys

sys.path.insert(1, str(Path(__file__).resolve().parents[3]))
from geometry_m2.training import main

if __name__ == "__main__":
    main("D1")
