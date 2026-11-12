"""Render fixed-camera panels from Our2 V3 example point clouds."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = (
    ("observed", "Input"),
    ("S0_midpoint_seed", "S0 Midpoint seeds"),
    ("S1_uniform_seed", "S1 uniform hard seeds"),
    ("S2_learned_seed", "S2 learned hard seeds"),
    ("target", "Ground truth"),
    ("S0_midpoint_prediction", "S0 final output"),
    ("S1_uniform_prediction", "S1 final output"),
    ("S2_learned_prediction", "S2 final output"),
)


def render(path, output, limits=None, panels=PANELS):
    with np.load(path, allow_pickle=False) as data:
        clouds = [data[key] for key, _ in panels]
        if any(cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all() for cloud in clouds):
            raise ValueError(f"Invalid example point clouds: {path}")
        if limits is None:
            stacked = np.concatenate(clouds)
            limits = stacked.min(0), stacked.max(0)
        low, high = limits
        center = (low + high) / 2
        radius = max(float((high - low).max()) * 0.53, 1e-5)
        figure = plt.figure(figsize=(12, 6.5), layout="constrained")
        for index, ((key, title), cloud) in enumerate(zip(panels, clouds), 1):
            ax = figure.add_subplot(2, 4, index, projection="3d")
            colors = "#275b79"
            if key == "observed":
                colors = np.where(data["outlier_mask"], "#ba4027", colors)
            ax.scatter(*cloud.T, c=colors, s=1.5, linewidths=0, alpha=.78, depthshade=False)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=45)
            ax.set_proj_type("ortho")
            ax.set_axis_off()
            ax.set_title(title)
        figure.suptitle(f"{data['condition']} | {data['object_id']}", fontsize=11)
        figure.savefig(output, dpi=150, facecolor="white")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic_root", type=Path, help="The v2_v3 directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.diagnostic_root.glob("*/examples/*.npz"))
    if not paths:
        parser.error("No V3 example .npz files found")
    bounds, keys = {}, {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            key = path.parents[1].name, str(data["object_id"])
            clouds = np.concatenate([data[name] for name, _ in PANELS])
            low, high = clouds.min(0), clouds.max(0)
        if key in bounds:
            low, high = np.minimum(low, bounds[key][0]), np.maximum(high, bounds[key][1])
        bounds[key] = low, high
        keys[path] = key
    for path in paths:
        destination = args.output / path.parents[1].name / f"{path.stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        render(path, destination, bounds[keys[path]])
    print(f"Rendered {len(paths)} fixed-camera panels under {args.output}")


if __name__ == "__main__":
    main()
