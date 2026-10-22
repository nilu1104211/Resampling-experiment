"""Run integration acceptance and create compact multi-view QA artifacts."""
from __future__ import annotations

import argparse
from io import StringIO
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import unittest
import zipfile

import h5py
import numpy as np

from .core import array_hash
from .pipeline import file_hash, read_json, write_json, log
from .v3_core import CONDITIONS
from .v3_dataset import AdaptiveResamplingDataset, EpochSampler, OfficialPU1KDataset, inverse_prediction
from .v3_metrics import evaluate_sample
from .v3_pipeline import DEFAULT, read_row


def worker_read(task):
    dataset, index = task
    sample = dataset[index]
    result = {"hash": array_hash(sample["input_xyz"]), "shape": sample["input_xyz"].shape}
    dataset.close()
    return result


def integration(root):
    dataset = AdaptiveResamplingDataset(root, "train", include_supervision=True, include_geometry_features=True)
    rng = np.random.default_rng(20261021)
    indices = rng.choice(len(dataset), min(100, len(dataset)), replace=False)
    checked, conditions = 0, set()
    start = time.perf_counter()
    for index in indices:
        for epoch in ([0, 1, 2, 3, 4] if checked < 30 else [int(index) % 5]):
            sample = dataset[(epoch, int(index))]
            repeated = dataset[(epoch, int(index))]
            if not np.array_equal(sample["input_xyz"], repeated["input_xyz"]):
                raise ValueError("Online sample is not reproducible")
            labels = sample["supervision"]
            if sample["input_features"].shape != (256, 8):
                raise ValueError("Observable geometry feature interface mismatch")
            if not np.allclose(sample["input_xyz"] + labels["denoise_offset"], labels["clean_input_xyz"], atol=5e-7):
                raise ValueError("Normalized/rotated denoising correspondence mismatch")
            canonical_target = inverse_prediction(sample["gt_xyz"], sample)
            raw = dataset.raw_sample(int(index), epoch)
            if not np.allclose(canonical_target, raw["gt_xyz"], atol=5e-7):
                raise ValueError("Prediction inverse does not recover benchmark frame")
            conditions.add(sample["condition"])
            checked += 1
    seconds = time.perf_counter() - start
    # Exercise spawn serialization after the parent has opened HDF5 handles.
    dataset[(3, 0)]
    tasks = [(dataset, (0, 0)), (dataset, (3, 0)), (dataset, (4, min(100, len(dataset) - 1)))]
    expected = [worker_read((dataset, item[1])) for item in tasks]
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        actual = pool.map(worker_read, tasks)
    if actual != expected:
        raise ValueError("Spawn workers do not reproduce parent samples/epochs")
    fixed = AdaptiveResamplingDataset(root, "pu1k_patch", condition="E2_mixed_outliers",
                                      include_supervision=True, include_reference_metadata=True)
    sample = fixed[0]
    if not np.array_equal(fixed[(0, 0)]["input_xyz"], fixed[(1234, 0)]["input_xyz"]):
        raise ValueError("Fixed evaluation changes with epoch")
    identity = evaluate_sample(sample["gt_xyz"], sample, include_p2f=True)
    if identity["cd_l1"] > 1e-6 or identity["p2f_max_l2"] > 2e-5:
        raise ValueError("Reference mesh/point evaluation self-check failed")
    official = OfficialPU1KDataset(root, include_reference_metadata=True)
    normal = official[0]
    raw_target = inverse_prediction(normal["gt_xyz"], normal, raw_coordinates=True)
    original = np.loadtxt(official.entries[0]["target_path"], dtype="f8")[:, :3]
    relative_error = float(np.max(np.abs(raw_target - original)) / normal["canonical_scale"])
    if relative_error > 2e-6 or len(official) != 127:
        raise ValueError("Official point pair adapter changed target coordinates")
    dataset.close()
    fixed.close()
    report = {"status": "passed", "online_epoch_samples_checked": checked, "online_condition_families": sorted(conditions),
              "spawn_workers": 2, "spawn_epoch_and_HDF5_transport": "passed", "fixed_evaluation_epoch_invariant": True,
              "observable_features": [256, 8], "identity_prediction_metrics": identity,
              "official_benchmark_pairs": len(official), "official_target_inverse_relative_error": relative_error,
              "serial_online_samples_per_second_with_features_and_replay": checked / seconds,
              "pytorch_DataLoader_executed": False, "pytorch_note": "Data preparation interpreter has no torch; actual spawn/HDF5 transport was tested"}
    torch_record = root / "checks/pytorch_loader_acceptance.json"
    if torch_record.exists() and read_json(torch_record)["status"] == "passed":
        report["pytorch_DataLoader_executed"] = True
        report["pytorch_note"] = "Actual persistent DataLoader workers also passed in the existing pufm environment"
        report["pytorch_environment"] = read_json(torch_record)
    write_json(root / "checks/integration_acceptance.json", report)
    return report


def previews(root, group="pu1k_patch"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    integrity = root / "checks" / ("preview_integrity.json" if group == "pu1k_patch" else f"preview_integrity_{group}.json")
    plotter_hash = file_hash(Path(__file__))
    if integrity.exists():
        previous = read_json(integrity)
        if previous.get("plotter_sha256") == plotter_hash and all(Path(p).exists() for p in previous["files"]):
            for path in previous["files"]:
                with Image.open(path) as image:
                    image.verify()
            return previous["files"]

    folder = root / "visualization"
    folder.mkdir(exist_ok=True)
    manifest = read_json(root / "metadata/base_manifest.json")
    entries = manifest["groups"][group]
    selected = [entries[i] for i in np.linspace(0, len(entries) - 1, 3, dtype=int)]
    names = ["A0_normal", "B3_sparse_heavy", "C3_hole_20", "D7_outliers_02", "D8_laplace_005_unseen", "E2_mixed_outliers"]
    result = []
    for condition in names:
        figure = plt.figure(figsize=(12, 10))
        for row, entry in enumerate(selected):
            path = root / "processed" / group / Path(entry["path"]).name
            with h5py.File(path, "r") as handle:
                sample = read_row(handle, list(CONDITIONS).index(condition))
            x, gt = sample["input_xyz"], sample["gt_xyz"]
            noisy = sample["noise_mask_input"].astype(bool)
            missing = sample["hole_mask_gt"].astype(bool)
            for column in range(3):
                axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d" if column == 0 else None)
                if column == 0:
                    axis.scatter(*gt.T, s=1, c="#b9c3ce", alpha=0.25)
                    if missing.any():
                        axis.scatter(*gt[missing].T, s=3, c="#efa93c", alpha=0.8)
                    axis.scatter(*x[~noisy].T, s=4, c="#1467a5", alpha=0.9)
                    if noisy.any():
                        axis.scatter(*x[noisy].T, s=5, c="#cf3d49", alpha=0.95)
                    extent = np.max(np.ptp(np.concatenate([x, gt]), axis=0)) / 2
                    middle = (np.max(gt, axis=0) + np.min(gt, axis=0)) / 2
                    axis.set_xlim(middle[0] - extent, middle[0] + extent)
                    axis.set_ylim(middle[1] - extent, middle[1] + extent)
                    axis.set_zlim(middle[2] - extent, middle[2] + extent)
                    axis.set_box_aspect((1, 1, 1))
                    axis.view_init(24, -55)
                else:
                    dimensions = [0, 1] if column == 1 else [0, 2]
                    axis.scatter(*gt[:, dimensions].T, s=2, c="#b9c3ce", alpha=0.4)
                    if missing.any():
                        axis.scatter(*gt[missing][:, dimensions].T, s=5, c="#efa93c", alpha=0.8)
                    axis.scatter(*x[~noisy][:, dimensions].T, s=7, c="#1467a5")
                    if noisy.any():
                        axis.scatter(*x[noisy][:, dimensions].T, s=9, c="#cf3d49")
                    axis.set_aspect("equal", adjustable="box")
                axis.set_title(["3D", "XY", "XZ"][column] + f" / object {row + 1}", fontsize=9)
                axis.tick_params(labelsize=6)
        figure.suptitle(group + " / " + condition + " | blue: observed / red: perturbed / amber: missing reference", fontsize=10)
        figure.tight_layout(rect=[0, 0, 1, 0.96])
        path = folder / (("" if group == "pu1k_patch" else group + "_") + condition + ".png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        with Image.open(path) as image:
            image.verify()
        result.append(str(path))
    write_json(integrity, {"status": "passed", "files": result, "examples_per_condition": 3,
                                                       "views_per_example": 3, "plotter_sha256": plotter_hash})
    return result


def unit_acceptance(root):
    project = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(project / "tests"))
    output = StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    record = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "log": output.getvalue()}
    write_json(root / "checks/unit_tests.json", record)
    if not result.wasSuccessful():
        raise ValueError("Unit acceptance failed")
    return record


def report(root):
    root = Path(root)
    validation = read_json(root / "checks/full_validation.json")
    if validation["status"] != "passed":
        raise ValueError("Full dataset validation has not passed")
    if read_json(root / "checks/contract_validation.json")["status"] != "passed":
        raise ValueError("Field/manifest contract validation has not passed")
    tests = unit_acceptance(root)
    accepted = integration(root)
    images = previews(root)
    images += previews(root, "modelnet40_patch")
    project = Path(__file__).resolve().parents[1]
    extras = [project / "tests/test_v3_data.py", project / "tests/test_data_pipeline.py", project / "examples/use_v3_data.py",
              project / "数据使用说明_V3.md", project / "README.md", project / "requirements_data_v3.txt",
              project / "tools/evaluate_v3_predictions.py", project / "tools/check_v3_torch_loader.py"]
    source_paths = sorted((project / "adaptive_resampling").glob("*.py")) + extras
    write_json(root / "metadata/code_fingerprints.json", {str(p.relative_to(project)): file_hash(p) for p in source_paths})
    with zipfile.ZipFile(root / "metadata/source_code_v3.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        # Include the shared V2 geometry primitives used by V3.
        for path in source_paths:
            archive.write(path, path.relative_to(project))
    write_json(root / "metadata/environment.json", {"executable": sys.executable, "python": sys.version,
                                                    "platform": platform.platform(), "numpy": np.__version__, "h5py": h5py.__version__})
    shutil_path = root / "数据使用说明_V3.md"
    shutil_path.write_text((project / "数据使用说明_V3.md").read_text(encoding="utf-8"), encoding="utf-8")
    files = list(root.rglob("*.h5"))
    summary = {"status": "computation_passed_visual_review_pending", "version": "3.0.0", "hdf5_files": len(files),
               "hdf5_GiB": sum(p.stat().st_size for p in files) / 1024**3,
               "base_groups": validation["base_groups"], "fixed_groups": validation["fixed_groups"],
               "unit_tests_passed": tests["tests_run"], "integration_passed": accepted["status"] == "passed",
               "field_contracts_passed": True, "actual_pytorch_dataloader_passed": accepted["pytorch_DataLoader_executed"],
               "preview_images": images, "network_training_started": False}
    write_json(root / "checks/completion_summary.json", summary)
    clean_counts = {item["group"]: item for item in validation["base_groups"]}
    fixed_counts = {item["group"]: item["rows_checked"] for item in validation["fixed_groups"]}
    lines = ["# 自适应点云重采样数据 V3", "", "计算验收通过；最终视觉复核状态见 checks/completion_summary.json。", "",
             "PU1K 主训练，ModelNet40 跨域测试。真实物体级划分，独立输入候选点与 GT，在线训练退化、固定验证测试。", "",
             "| 分组 | 物体数 | 基础点云 | 固定评测记录 |", "|---|---:|---:|---:|"]
    for name, item in clean_counts.items():
        lines.append(f"| {name} | {item['objects']} | {item['clean_bases_checked']:,} | {fixed_counts.get(name, 0):,} |")
    lines += ["", f"共 {len(files):,} 个 HDF5，占 {summary['hdf5_GiB']:.2f} GiB。训练基础点云随 epoch 生成不同退化，不能把基础点云数量当成固定退化总数。", "",
              "局部输入/输出为 256/1024；整物体为 2048/8192。17 种固定测试条件包含干净、非均匀、缺失、高斯比例变化、离群点、混合及训练未见过的 Laplace 噪声。", "",
              "使用与命令见 数据使用说明_V3.md。Python 接口为 adaptive_resampling.v3_dataset.AdaptiveResamplingDataset。", "",
              "checks/ 保存逐条几何验收、字段协议、数值测试、真实 PyTorch worker 检查与可视化证据；metadata/ 保存物体划分、原始网格路径、配置和源码包。", "",
              "新版从公开 900 个训练网格重新采样，不等同于旧 addpugan HDF5 训练协议。论文主对比需要在同一新版数据上重训各网络。"]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("V3 numerical acceptance passed; preview images are ready for visual review")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT["output_root"])
    args = parser.parse_args()
    report(args.root)
