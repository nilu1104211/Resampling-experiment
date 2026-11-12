"""Aggregate verified outputs and make the twenty diagnostic questions auditable."""

from collections import defaultdict
from statistics import mean

from our2_p0 import CONDITIONS, MAIN_METRICS, read_json, read_rows, run, write_rows
from our2_diagnosis_seeds import RANDOM_GROUPS
from our2_v56 import quality_failures


METRICS = (*MAIN_METRICS, "pred_to_gt_mean_l2")
HOLE = ("C2_hole_10", "E1_mixed", "E2_mixed_outliers")


def numbers(rows, field):
    return [float(r[field]) for r in rows if r.get(field) not in ("", None)]


def average(rows, field):
    values = numbers(rows, field)
    return mean(values) if values else None


def describe(value):
    return f"{value:.6g}" if value is not None else "NA"


def metric_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scope"], row["split"], row["group"], row["checkpoint_sha256"]].append(row)
    result, summaries = [], {}
    for (scope, split, group, sha), records in sorted(grouped.items()):
        entries = [{"condition": r["condition"], "object_id": r["object_id"],
                    **{key: float(r[key]) for key in METRICS if r.get(key) not in ("", None)}} for r in records]
        present = sorted({r["condition"] for r in entries})
        if len(present) == len(CONDITIONS):
            summary, _ = run.aggregate(entries)
        else:
            # Partial and fixed examples still have useful condition results.
            by_condition = {condition: [r for r in entries if r["condition"] == condition] for condition in present}
            detail = {c: {m: average(items, m) for m in METRICS} for c, items in by_condition.items()}
            summary = {"conditions": detail, "macro": {m: average(list(detail.values()), m) for m in METRICS}}
        summary["complete_condition_matrix"] = len(present) == len(CONDITIONS)
        summaries[scope, split, group] = summary
        for condition, metrics in [("macro", summary["macro"]), *summary["conditions"].items()]:
            selected = [r for r in records if condition == "macro" or r["condition"] == condition]
            result.append({"scope": scope, "split": split, "group": group, "checkpoint_sha256": sha,
                           "condition": condition, "samples": len(selected),
                           "objects": len({r["object_id"] for r in selected}),
                           "conditions_present": len(present), **{m: metrics.get(m) for m in METRICS}})
    return result, summaries


def paired(rows, group, reference, allowed=None):
    """Pair by split, scope, condition, sample ID, exact input and exact GT."""
    key = lambda r: (r["scope"], r["split"], r["condition"], r["sample_id"],
                     r["input_sha256"], r["gt_sha256"])
    left = {key(r): r for r in rows if r["group"] == group}
    right = {key(r): r for r in rows if r["group"] == reference}
    output = []
    for pair in sorted(left.keys() & right.keys()):
        if allowed is not None and (pair[1], pair[3]) not in allowed:
            continue
        a, b = left[pair], right[pair]
        output.append({"scope": pair[0], "split": pair[1], "condition": pair[2], "sample_id": pair[3],
                       "object_id": a["object_id"], "input_sha256": pair[4], "gt_sha256": pair[5],
                       "group": group, "reference": reference, "checkpoint_sha256": a["checkpoint_sha256"],
                       "reference_checkpoint_sha256": b["checkpoint_sha256"],
                       **{m: float(a[m]) - float(b[m]) for m in METRICS
                          if a.get(m) not in ("", None) and b.get(m) not in ("", None)}})
    return output


def pair_summary(rows):
    result = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scope"], row["split"], row["group"], row["reference"]].append(row)
    for (scope, split, group, control), values in sorted(grouped.items()):
        conditions = sorted({r["condition"] for r in values})
        for condition in ["macro", *conditions]:
            selected = [r for r in values if condition == "macro" or r["condition"] == condition]
            blocks = defaultdict(list)
            for row in selected:
                blocks[row["condition"], row["object_id"]].append(row)
            for metric in METRICS:
                per_condition = defaultdict(list)
                per_object = defaultdict(list)
                for (c, obj), items in blocks.items():
                    delta = average(items, metric)
                    if delta is not None:
                        per_condition[c].append(delta)
                        per_object[obj].append(delta)
                if not per_condition:
                    continue
                sign = 1 if metric in ("f_score", "precision", "recall", "hole_coverage_fraction",
                                        "sparse_coverage_fraction") else -1
                effects = [sign * mean(items) for items in per_object.values()]
                result.append({"scope": scope, "split": split, "group": group, "reference": control,
                               "condition": condition, "metric": metric,
                               "delta": mean(mean(items) for items in per_condition.values()),
                               "paired_samples": len(numbers(selected, metric)), "paired_objects": len(effects),
                               "objects_improved": sum(d > 0 for d in effects),
                               "objects_worse": sum(d < 0 for d in effects),
                               "checkpoint_sha256": values[0]["checkpoint_sha256"],
                               "reference_checkpoint_sha256": values[0]["reference_checkpoint_sha256"],
                               "example_sample_id": selected[0]["sample_id"]})
    return result


def historical(args):
    if args.debug:
        return []
    rows = []
    provenance = read_json(args.output / "manifest.json")
    for split in ("validation", "pu1k_object", "modelnet40_object"):
        path = args.p0_results / "v2_v3" / split / "summary.json"
        if not path.is_file():
            continue
        summary = read_json(path)
        identity = summary["identity"]
        for key, expected in (("baseline_sha256", args.checkpoint_info["B0"]["sha256"]),
                              ("metadata_sha256", provenance["data_metadata_sha256"]),
                              ("training_source_sha256", provenance["training_source_sha256"]),
                              ("runtime", provenance["runtime"])):
            if identity[key] != expected:
                raise ValueError(f"Historical V2/V3 {key} differs for {split}")
        for group, group_result in summary["groups"].items():
            for condition, metrics in [("macro", group_result["macro"]), *group_result["conditions"].items()]:
                count = group_result["samples"] if condition == "macro" else group_result["samples"] // 7
                rows.append({"scope": "historical_v2_v3", "split": split, "group": group,
                             "checkpoint_sha256": args.checkpoint_info["B0"]["sha256"],
                             "condition": condition, "samples": count, "objects": group_result["objects"],
                             **{metric: metrics.get(metric) for metric in METRICS}, "source": str(path)})
    for split in ("pu1k_object", "modelnet40_object"):
        for group, stage in (("D0", "baseline"), ("M1-full", "m1_refine"), ("M2", "m2_hard"),
                             ("J3", "joint_finetune"), ("B1", None)):
            path = (args.previous_eval / split / stage / "summary.json" if stage
                    else args.p0_results / "v1_extra5" / split / "summary.json")
            if not path.is_file():
                continue
            summary = read_json(path)
            model = {"D0": "B0", "M1-full": "M1", "J3": "Joint"}.get(group, group)
            digest = args.checkpoint_info[model]["sha256"]
            if (summary["checkpoint_sha256"] or summary["baseline_sha256"]) != digest or any(
                summary[key] != provenance[other] for key, other in
                (("metadata_sha256", "data_metadata_sha256"), ("code_sha256", "training_source_sha256"),
                 ("runtime", "runtime"))
            ):
                raise ValueError(f"Historical whole-object identity differs: {path}")
            for condition, metrics in [("macro", summary["macro"]), *summary["conditions"].items()]:
                rows.append({"scope": "historical_whole", "split": split, "group": group,
                             "checkpoint_sha256": digest, "condition": condition,
                             "samples": summary["samples"] if condition == "macro" else summary["samples"] // 7,
                             "objects": summary["objects"], **{metric: metrics.get(metric) for metric in METRICS},
                             "source": str(path)})
    return rows


def conclusions(evidence, pairs, decision):
    patch = [r for r in pairs if r["scope"] == "patch" and r["condition"] == "macro"]
    lookup = {(r["group"], r["reference"], r["metric"]): r for r in patch}
    heads = [r for r in evidence["m1_heads"] if r["scope"] == "patch" and r["flow"] == "M1"]
    shift = [r for r in evidence["m1_seed_flow"] if r["group"] == "M1-full" and r["scope"] == "patch"]
    start = [r for r in shift if r["step"] == "0"]
    joint = [r for r in evidence["joint_interaction"] if r["group"] == "J4" and r["scope"] == "patch"]
    moves = [r for r in evidence["matched_moves"] if r["group"] == "D1-L" and r["scope"] == "patch"]
    proxies = evidence["proxy_gain"]
    probes = evidence["gradient_training"]
    relative_stages = {
        step: (average([r for r in shift if r["step"] == str(step)], "hole_gt_to_seed"),
               average([r for r in evidence["m1_seed_flow"] if r["scope"] == "patch"
                        and r["group"] == "M1-off" and r["step"] == str(step)], "hole_gt_to_seed"))
        for step in range(6)
    }
    seed_a, seed_b = relative_stages[0]
    step_deltas = ", ".join(f"t={step / 5:g}: {describe(a - b if a is not None and b is not None else None)}"
                            for step, (a, b) in relative_stages.items())
    gradient_outputs = [r for r in probes if r.get("update_l2") not in ("", None)]
    groups_with_updates = sorted({r["parameter_block"] for r in gradient_outputs if float(r["update_l2"]) > 0})

    def comparison(group, control, metric):
        row = lookup.get((group, control, metric))
        if row is None:
            return "无可用配对。"
        return (f"{group} − {control} 的 {metric} 为 {row['delta']:+.6g}，"
                f"{row['paired_samples']} 个配对输入，{row['paired_objects']} 个物体，"
                f"示例 `{row['example_sample_id']}`。")

    def head(label, field, name):
        found = [r for r in heads if r.get("label") == label]
        return f"{name} {describe(average(found, field))}（样本等权，{len(numbers(found, field))} 个有效输入）"

    def positive_calls(label):
        found = [r for r in heads if r.get("label") == label]
        tp = int(sum(numbers(found, "tp")))
        total = tp + int(sum(numbers(found, "fn")))
        return f"{tp}/{total}" if total else "NA"

    def direction_status():
        comparisons = [lookup.get(("D1-L", other, "f_score")) for other in ("D1-U", *RANDOM_GROUPS)]
        if any(row is None for row in comparisons):
            return "证据不足"
        if all(row["delta"] > 0 for row in comparisons):
            return "支持"
        return "否定" if all(row["delta"] <= 0 for row in comparisons) else "证据不足"

    def orientation(a, b, higher=True):
        if a is None or b is None:
            return "证据不足"
        return "支持" if (a > b if higher else a < b) else "否定"

    def average_head(label, field):
        return average([r for r in heads if r.get("label") == label], field)

    noise = average_head("noise", "auroc")
    outlier = average_head("outlier", "auroc")
    h_auc = average_head("training_soft_target_at_0.5", "auroc")
    direction = direction_status()
    if decision["d1_quality_failures"] and direction == "支持":
        next_step = "学习方向在配对 F-score 有正增益，D1-L 仍未过 D0 质量门槛。先查候选与删补的几何损失。"
    elif direction == "否定":
        next_step = "学习方向的配对 F-score 未优于各控制，先看需求代理与离线增点收益的相关性。"
    else:
        next_step = "配对方向结论待定。结合局部距离和训练对照，分别查候选几何、预算监督与 Flow。"

    rows = [
        ("A02", "r 识别纯噪声和离群点",
         orientation(min(noise, outlier), .5) if noise is not None and outlier is not None else "证据不足",
         f"阈值 0.5 下识别出的正例/标签正例，噪声 {positive_calls('noise')}，离群 {positive_calls('outlier')}。"
         + head("noise", "auroc", "噪声 AUROC") + "，" + head("outlier", "auroc", "离群 AUROC") + "。", "m1_heads.csv"),
        ("A02", "r 是否误判自然边缘", "证据不足", "现有标签可查离群和腐蚀噪声，自然边缘没有独立真值。", "m1_heads.csv"),
        ("A02", "h 是否定位缺失", orientation(h_auc, .5),
         head("training_soft_target_at_0.5", "recall", "候选召回") + "，"
         + head("training_soft_target_at_0.5", "auroc", "候选 AUROC")
         + "。正例按原训练软标签 0.5 阈值定义。", "m1_heads.csv"),
        ("A02", "h 是否误判自然开口或薄壁", "证据不足", "需要开口、薄壁和自然边界独立标注。A0 是全局负例。", "m1_heads.csv"),
        ("A03/C03", "M1 位移为何偏小", "证据不足",
         f"full 位移 / spacing 均值 {describe(average(start, 'shift_over_spacing'))}，"
         f"gamma.tanh {describe(average(start, 'gamma_tanh'))}，"
         f"接近裁剪上限的种子比例 {describe(average(start, 'movement_cap_fraction'))}。", "m1_seed_flow.csv"),
        ("A03", "M1 种子能否改善孔洞覆盖", orientation(seed_a, seed_b, higher=False),
         f"相同输入下 step=0 平均孔洞距离，full {describe(seed_a)}，off {describe(seed_b)}。", "m1_seed_flow.csv"),
        ("A03", "Flow 各步如何改变 M1 修正", "证据不足" if seed_a is None else "见轨迹",
         f"网络坐标中各步 full − off 孔洞距离差，{step_deltas}。"
         + comparison("M1-full", "M1-off", "hole_coverage_mean_l2"),
         "m1_seed_flow.csv"),
        ("C01", "匹配训练能否拆出 M1 生成作用", "见指标",
         comparison("T2", "T1", "hole_coverage_mean_l2") + comparison("T1", "T0", "hole_coverage_mean_l2"), "paired_summary.csv"),
        ("B03", "预算是否随输入变化", "见指标",
         f"D1-L 实际替换均值 {describe(average(moves, 'actual_m'))}，按条件和区域核对目标及配额变化。", "m2_budgets.csv"),
        ("B02", "同等替换量下学习方向是否更好", direction_status(),
         " ".join(comparison("D1-L", control, "f_score") for control in ("D1-U", *RANDOM_GROUPS)), "paired_summary.csv"),
        ("B03", "移走与新增点改变了哪些覆盖", "见指标",
         f"D1-L 平均有 {describe(average(moves, 'gt_points_nearer'))} 个 GT 点变近，"
         f"{describe(average(moves, 'gt_points_farther'))} 个 GT 点变远。区域变化见 m2_budgets.csv。", "matched_moves.csv"),
        ("B03/B05", "新增点是否更偏离表面",
         orientation(average(moves, "added_seed_to_gt"), average(moves, "removed_seed_to_gt")),
         f"D1-L 新增点到 GT 均距 {describe(average(moves, 'added_seed_to_gt'))}，"
         f"移走点到 GT 均距 {describe(average(moves, 'removed_seed_to_gt'))}。", "matched_moves.csv"),
        ("B04", "PUFM 坐标、索引与邻域是否错配", "见指标",
         "配额实现检查每个输入的修改槽位与实际区域预算。现有 Flow 每步从 query XYZ 重算 KNN 和采样，"
         "接口没有外传逐点特征或冻结的邻域表。", "tensor_contract.json"),
        ("B06", "需求得分是否对应增点收益", "见指标",
         f"固定样本的区域边际收益 Spearman 均值 {describe(average(proxies, 'within_sample_spearman'))}。"
         "每次增一点均从固定供体区域减一点。", "proxy_gain.csv"),
        ("C02", "冻结及微调能保留多少 M2 质量", "见指标",
         comparison("D1-L-FT", "D1-L", "f_score") + comparison("D1-U-FT", "D1-U", "f_score")
         + comparison("D1-L-FT", "D1-U-FT", "f_score"), "paired_summary.csv"),
        ("D01", "Joint 的条件输入怎样改变配额", "见指标",
         f"J3/J4 的 logit L2 均值 {describe(average(joint, 'logit_l2_vs_j3'))}，"
         f"预算 L1 均值 {describe(average(joint, 'budget_l1_vs_j3'))}。"
         + comparison("J3", "J4", "f_score"), "joint_interaction.csv"),
        ("D02", "Joint 最早在哪一阶段变差", "见指标",
         "对照 Midpoint、分配后与修正后种子，再看五步 Flow；不同 checkpoint 的模块组仅作结果参照。"
         + comparison("J3", "J2", "f_score"), "joint_seed_flow.csv"),
        ("C03/D03", "模块梯度和参数实际更新", "支持" if groups_with_updates else "证据不足",
         f"记录 {len(probes)} 条探针。观察到非零参数更新的块为 {', '.join(groups_with_updates) or 'NA'}。"
         "探针表还列出各损失在共享参数上的梯度与余弦。", "gradient_training.csv"),
        ("E01", "Patch 和整物体表现是否一致", "证据不足",
         "按 scope 比较；固定案例不能代表 80 个物体。整物体关键组全量取决于 --whole-regression。", "comparison_all.csv"),
        ("E02", "下一步应该优先查哪个环节", "见指标",
         f"D1-L 质量门槛失败数 {len(decision['d1_quality_failures'])}。{next_step}", "decision.json"),
    ]
    return [{"question": i, "experiment": test, "topic": topic, "status": status,
             "detail": detail, "evidence": file} for i, (test, topic, status, detail, file) in enumerate(rows, 1)]


def report(args, queue, folder):
    folder.mkdir(parents=True, exist_ok=True)
    evidence = defaultdict(list)
    for task in queue.state["tasks"].values():
        if task["status"] != "completed":
            continue
        for path in task["files"]:
            file = args.output / path
            if file.suffix == ".csv" and file.name not in ("train.csv",):
                evidence[file.stem].extend(read_rows(file))
    rows = evidence.pop("per_sample")
    rows.extend({**r, "group": "D0-FT"} for r in list(rows) if r["group"] == "T0")
    comparison, summaries = metric_rows(rows)
    try:
        comparison += historical(args)
    except (KeyError, ValueError) as exc:
        queue.failures["historical_comparison"] = {"reason": "INVALID", "message": str(exc)}
    matches = defaultdict(set)
    for row in evidence["matched_moves"]:
        if row["matched"] == "True" and int(row["actual_m"]) > 0:
            matches[row["group"]].add((row["split"], row["sample_id"]))
    pairs = []
    for group, control in (
        ("M1-full", "M1-off"), ("M1-r", "M1-full"), ("M1-h", "M1-full"), ("M1-full", "B1"),
        ("D1-L", "D0"), ("J3", "J4"), ("J3", "J2"), ("J1", "J0"), ("J2", "J0"), ("J3", "J0"),
        ("T2", "T1"), ("T1", "T0"), ("T2", "T0"), ("D1-L-FT", "D1-L"), ("D1-U-FT", "D1-U"),
        ("D1-L-FT", "D1-U-FT"), ("D1-L-FT", "D0-FT"), ("D1-U-FT", "D0-FT"),
    ):
        pairs.extend(paired(rows, group, control))
    for control in ("D1-U", *RANDOM_GROUPS):
        pairs.extend(paired(rows, "D1-L", control, matches[control]))
    pair_metrics = pair_summary(pairs)
    a, b = (summaries.get(("patch", "validation", g)) for g in ("D1-L", "D0"))
    decision = {"d1_quality_failures": quality_failures(a, b) if a and b and a["complete_condition_matrix"]
                and b["complete_condition_matrix"] else [], "quality_gate_evaluated": bool(
                    a and b and a["complete_condition_matrix"] and b["complete_condition_matrix"] and not args.debug),
                "nonzero_matched": {g: len(m) for g, m in matches.items()}, "debug": bool(args.debug)}
    verdicts = conclusions(evidence, pair_metrics, decision)
    if args.debug:
        for item in verdicts:
            item["status"] = "短测"
    if not rows:
        for item in verdicts:
            item["status"] = "证据不足"
    output = []
    for name, content in {**evidence, "comparison_all": comparison, "sample_diagnostics": rows,
                          "paired_samples": pairs, "paired_summary": pair_metrics}.items():
        if content:
            path = folder / f"{name}.csv"
            write_rows(path, content)
            output.append(path)
    contract = {"query": "B,N,3 -> B,3,N", "condition_tensor": "B,3,N; passed but unused by PUFM_w_attn.forward",
                "knn": "Recomputed from the current query coordinates at each Euler step",
                "seed_slots": "D1 preserves all unmodified Midpoint slots and logs replacements",
                "input_channels": ["x", "y", "z"], "flow_times": [0, .2, .4, .6, .8, 1],
                "gt_forward": False}
    for name, payload in (("conclusions", verdicts), ("decision", decision), ("tensor_contract", contract)):
        path = folder / f"{name}.json"
        run.write_json(path, payload)
        output.append(path)
    lines = ["# Our2 / PUFM 诊断", "", f"有效输出行数 {len(rows)}。配对行数 {len(pairs)}。",
             f"队列异常或阻断 {len(queue.failures)} 项，见 `../failures.json`。", ""]
    if args.debug:
        lines.extend(["**短测。仅验证数据、训练和产物路径。**", ""])
    lines.extend(["权重与运行环境见 `../manifest.json`。配对表记录双方 SHA256、条件、样本数和示例输入。",
                  "整物体固定案例及全量回归以 scope 区分。每一组训练默认报告相同更新预算下的 last 权重。", ""])
    for item in verdicts:
        lines.extend([f"## {item['question']}. {item['topic']} · {item['experiment']}", "",
                      f"**{item['status']}**。{item['detail']} 证据见 `{item['evidence']}`。", ""])
    path = folder / "diagnosis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return [*output, path]
