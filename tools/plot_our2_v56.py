"""Render fixed-camera point-cloud comparisons for the V5 and V6 examples."""

import argparse
from pathlib import Path

import numpy as np

from plot_our2_p0 import render


def panels_for(keys):
    common = (("observed", "Input"), ("original_seed", "Original Midpoint"))
    if "M1-full_seed" in keys:
        yield "M1", (*common, ("M1-full_seed", "M1 full seeds"), ("M1-off_seed", "M1 off seeds"),
                     ("target", "Ground truth"), ("M1-full_prediction", "M1 full output"),
                     ("M1-off_prediction", "M1 off output"))
    for group in ("D1", "D2", "D3"):
        if f"{group}-L_seed" in keys and f"{group}-U_seed" in keys:
            yield group, (*common, (f"{group}-L_seed", f"{group}-L seeds"), (f"{group}-U_seed", f"{group}-U seeds"),
                          ("target", "Ground truth"), ("D0_prediction", "D0 output"),
                          (f"{group}-L_prediction", f"{group}-L output"), (f"{group}-U_prediction", f"{group}-U output"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks, bounds = [], {}
    for path in sorted(args.result_root.glob("v[56]/*/*/examples/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for group, panels in panels_for(data.files):
                key = path.parents[1].relative_to(args.result_root), str(data["object_id"])
                cloud = np.concatenate([data[name] for name, _ in panels])
                lo, hi = cloud.min(0), cloud.max(0)
                if key in bounds:
                    lo, hi = np.minimum(lo, bounds[key][0]), np.maximum(hi, bounds[key][1])
                bounds[key] = lo, hi
                tasks.append((path, key, group, panels))
    if not tasks:
        parser.error("No V5/V6 examples found")
    for path, key, group, panels in tasks:
        destination = args.output / key[0] / f"{path.stem}_{group}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        render(path, destination, bounds[key], panels)
    print(f"Rendered {len(tasks)} fixed-camera V5/V6 panels under {args.output}")


if __name__ == "__main__":
    main()
