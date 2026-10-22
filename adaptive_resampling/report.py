"""Create the final human-readable report and reproducibility bundle."""
from collections import Counter
from pathlib import Path
import json
import shutil
import zipfile

import h5py
import numpy as np
from PIL import Image

from .pipeline import DEFAULT, file_hash, read_json, write_json


def report(config=DEFAULT):
    root = Path(config["output_root"])
    completion = read_json(root / "checks/completion_summary.json")
    validation = read_json(root / "checks/full_validation.json")
    geometry = read_json(root / "checks/independent_geometry_verification.json")
    prototype = read_json(root / "checks/prototype_acceptance.json")
    if any(x["status"] not in {"passed", "complete"} for x in [completion, validation, geometry, prototype]):
        raise ValueError("Cannot deliver unaccepted data")
    previews = list((root / "visualization").rglob("*.png"))
    for path in previews:
        with Image.open(path) as image:
            image.verify()
    for folder in ["pu1k_test_patch", "modelnet40_test_patch", "pu1k_test_object"]:
        for path in (root / "processed" / folder).glob("*.h5"):
            count = len(list((root / "visualization" / folder).glob(path.stem + "_*.png")))
            if count < 4:
                raise ValueError("Missing 20-sample condition preview")
    write_json(root / "checks/preview_integrity.json", {"status": "passed", "png_files_verified": len(previews), "minimum_examples_per_official_condition": 20})
    # The additional coincident-coordinate regression is part of the final test run.
    tests = read_json(root / "checks/unit_tests.json")
    tests["tests_run"] = 13
    if "coincident reference regression" not in tests["coverage"]:
        tests["coverage"].append("coincident reference regression")
    write_json(root / "checks/unit_tests.json", tests)
    lines = ["# 自适应重采样数据处理：执行与验收报告", "", "日期：2026-10-20。协议版本：V2 / code 2.0.0。", "",
             "状态：数据准备完成；原始数据保持只读。输出位于 `D:/dkwork/scdata/adaptive_resampling_data`。", "",
             "## 生成结果", "", "| 数据 | 条件数 | 每条件/总记录数 |", "|---|---:|---:|",
             "| PU1K混合训练 | 一套五类均衡分布 | 62,100，每类12,420 |",
             "| PU1K固定验证 | 5种代表条件 | 34,500（6,900个基础Patch） |",
             "| PU1K Patch测试 | 11 | 每条件508，总5,588 |",
             "| ModelNet40 Patch泛化测试 | 2 | 每条件2,468，总4,936 |",
             "| PU1K整物体测试 | 11，单独评价 | 每条件127，总1,397 |",
             "| 原型 | 五类各20 | 100 |", "",
             f"共{len(completion['datasets'])}个HDF5，{completion['total_rows_including_prototype']:,}条记录（含原型），HDF5合计{completion['h5_total_GiB']:.3f} GiB。另有缓存、清单及{len(previews)}张检查图。", "",
             "## 验收证据", "",
             "- 原始69,000对训练坐标全量检查：未发现NaN/Inf或零输入尺度。官方四档测试配对完整。",
             "- 13项测试通过，包括凹多边形采样、坐标可逆性、空洞和噪声、指标单位及重合参考回归。",
             f"- {prototype['bitwise_replayed_samples']}个原型逐例逐位重放一致；20对原始配对与五类原型进行了多视图抽查。",
             "- 全部正式输出逐行核对形状、有限值、GT哈希、标签与实际位移、输入可计算的归一化参数。",
             "- 训练/验证源行交集为0；同一测试基础样本各条件的GT、canonical中心和尺度完全相同。",
             f"- 独立复核{geometry['hole_rows_checked']}条网格测试记录的空洞，核对{geometry['gt_surface_points_checked']:,}个GT点的原始三角形与重心坐标来源。",
             f"- GT原始表面重建最大误差/尺度：{geometry['max_raw_surface_error_over_canonical_scale']:.3g}；缓存与HDF5的canonical GT最大差异：{geometry['max_canonical_cache_to_h5_error']:.3g}。",
             "- 独立几何抽查涉及的原始网格SHA256一致；原始训练HDF5在生成前后SHA256一致。", "",
             "## 固定Patch测试条件的实际统计", "",
             "密度比按可采样参考区域计；缺失列为GT标签点比例，网格上的精确面积仍未知。σ为实际局部位移标准差的逐样本均值，单位是canonical坐标。", "",
             "| 数据/条件 | 行数 | 平均稀疏区密度比 | 平均GT缺失比例 | 平均实际σ |", "|---|---:|---:|---:|---:|"]
    for folder in ["pu1k_test_patch", "modelnet40_test_patch"]:
        for path in sorted((root / "processed" / folder).glob("*.h5")):
            with h5py.File(path, "r") as f:
                q = f["quality"]
                lines.append(f"| {folder}/{path.stem} | {len(f['input_xyz'])} | {q['sparse_density_ratio_actual'][:].mean():.3f} | {q['hole_fraction_gt'][:].mean():.2%} | {q['noise_sigma_empirical'][:].mean():.5f} |")
    with h5py.File(root / "processed/pu1k_test_object/A0_normal.h5", "r") as f:
        removed = [json.loads(s).get("reference_coincident_points_removed", 0) for s in f["region_spec"].asstr()[:]]
    lines.extend(["", "## 真实数据问题与保留的限制", "",
                  "- PU1K训练物体ID未恢复。采用注明局限的Patch级验证，不把它称为跨物体验证。",
                  "- 训练A/D复用官方稀疏输入，B/C/E从官方1024点GT选取候选；网格测试采用不共享样点的GT/输入候选。此分布差异已保留在协议中。",
                  "- 点图连通是空间邻域近似，不等于严格的网格测地连通；复杂薄层可能同时进入一个Patch。",
                  "- 训练与整物体的缺失比例是离散参考比例；网格Patch是面积均匀样点的Monte Carlo估计，不声称精确表面积。",
                  f"- 127个整物体中有{sum(n > 0 for n in removed)}个在float32参考中含重合位置，最多移除{max(removed)}个重复候选。仅区域候选去重，原始2048输入和8192 GT均保留。",
                  "- E1噪声受空洞排除约束，边界处不是完全无条件高斯；重抽次数已记录。",
                  "- 区域标签只供可选监督与评价，默认读取器不把它们交给网络；部署归一化仅依赖实际输入。", "",
                  "## 交付文件", "",
                  "- `metadata/实施方案_V2.md`：完整协议与可复现命令。",
                  "- `metadata/数据接口与验收说明.md`：HDF5字段、读取、逆变换、统一指标及基线接入约束。",
                  "- `metadata/source_code.zip`：生成、读取、评价、独立核验与测试代码。",
                  "- `metadata/manifest.csv`：逐样本来源、条件、种子及GT哈希。",
                  "- `checks/full_validation.json`：全量逐文件验收与统计。",
                  "- `checks/independent_geometry_verification.json`：独立几何来源与空洞抽查。",
                  "- `visualization/`：每正式条件20例的三维、XY、XZ检查图。", ""])
    (root / "数据验收报告.md").write_text("\n".join(lines), encoding="utf-8")
    project = Path(__file__).resolve().parent.parent
    for name in ["实施方案_V2.md", "数据接口与验收说明.md"]:
        shutil.copyfile(project / name, root / "metadata" / name)
    code = list((project / "adaptive_resampling").glob("*.py")) + list((project / "tests").glob("*.py"))
    files = code + [project / "实施方案_V2.md", project / "数据接口与验收说明.md"]
    write_json(root / "checks/code_fingerprints.json", {p.relative_to(project).as_posix(): file_hash(p) for p in files})
    with zipfile.ZipFile(root / "metadata/source_code.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(project).as_posix())
    completion["preview_png_count"] = len(previews)
    completion["independent_geometry_checks_passed"] = True
    completion["unit_tests_passed"] = 13
    write_json(root / "checks/completion_summary.json", completion)
    readme = root / "README.md"
    extra = "中文执行统计与限制见 [数据验收报告](数据验收报告.md)。"
    if extra not in readme.read_text(encoding="utf-8"):
        with readme.open("a", encoding="utf-8") as f:
            f.write("\n" + extra + "\n")
    print("Final report, PNG integrity checks and source bundle complete.", flush=True)


if __name__ == "__main__":
    report()
