"""Use the existing ID-paired, canonical-coordinate V3 evaluator unchanged."""
from pathlib import Path
import os
import runpy

if __name__ == "__main__":
    root = Path(os.environ.get("V3_CODE_ROOT", "D:/dkwork/scdata"))
    runpy.run_path(str(root / "tools/evaluate_v3_predictions.py"), run_name="__main__")
