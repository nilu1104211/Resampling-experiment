# Our2 完整诊断运行说明

入口为 `tools/diagnose_our2_all.py`。默认使用 PUFM-60、原 Our2 训练队列、v3 数据和原有整物体研发回归名单。各个实验按同一队列顺序执行。V6 的质量门槛用于记录 D1-L 相对 D0 的损失，均匀和随机对照照常运行。

| 阶段 | 比较与产物 |
| --- | --- |
| 身份 | 原 PUFM 和包装器逐点等价；正式运行核对历史输入与参考预测哈希。每个新对照记录输入、GT 和预测 SHA256 |
| M1 | 完整及分支开关的 Patch 评估；可靠性和孔洞候选诊断；种子位移、门控、几何和五步 Flow |
| M2 | 固定 D1 的 `λ=0.25, ρmax=0.05`；每输入等实际替换数的 U 和三个固定随机方向（21、22、23）；来源/去向预算、离线一进一出代理收益和五步 Flow |
| Joint | 同一已训练 Joint Flow 上的 J0–J4；条件开关、硬分配前后、M1 修正及逐步结果 |
| 梯度 | 在 v3 训练集的七种在线退化分层中固定取 21 个样本；测真实损失、参数梯度、辅助与生成梯度方向和探针更新 |
| 匹配训练 | T0 / T1 / T2 和 D0-FT / D1-L-FT / D1-U-FT。T0 与 D0-FT 共用同一训练与评估；B1 经身份、配置和更新预算校验后可复用为 T0 |
| 整物体 | 默认生成 PU1K 和 ModelNet40 各两物体的 A0/C2/D7/E2 成对图与轨迹；`--whole-regression` 额外跑两个固定 80 物体名单的关键对照 |

M1 两个训练分支共用 `m1_hole/best.pt`。T1 在最后五轮关闭位移，只适配 Flow；T2 在相同窗口开启原 M1 修正与损失。M2 使用冻结的预算头，L 和 U 各自适配 Flow。为复用同一纯 Flow 对照，各组最后窗口沿用 M1 的第 70 轮数据偏移、seed 21、batch 8 和 Flow LR `1e-6`。训练保存 epoch 0、每轮验证、`best.pt`、`last.pt`，诊断配对使用相同更新预算下的 `last.pt`。已有头预热单独记录。

均匀目标容量不够时，程序优先沿原均匀方向扩大容量，再求可行分区。具体策略和真实替换数写入 `matched_moves.csv`，包括零改动样本。方向配对只使用非零等量输入。自然开口、薄壁和真实边缘缺乏独立真值，报告按证据不足处理。轨迹及预算代理使用网络归一化坐标，最终指标沿用 v3 canonical 坐标。

## 本机上传

在仓库的 PowerShell 中执行。提交限定在本轮文件，方便保留工作区中其他实验的修改。

```powershell
Set-Location 'D:\dkwork\baseline_compare'
$files = @(
  'tools/diagnose_our2_all.py'
  'tools/our2_diagnosis_inference.py'
  'tools/our2_diagnosis_report.py'
  'tools/our2_diagnosis_seeds.py'
  'tools/our2_diagnosis_training.py'
  'docs/our2_full_diagnosis_runbook_2026-11-12.md'
)
git add -- $files
git diff --cached --stat -- $files
git commit -m "Add complete Our2 diagnosis queue" -- $files
if ($LASTEXITCODE -eq 0) { git push origin main }
```

## 服务器启动

在服务器 Bash 中执行。命令使用 `scdata-v3` 的原 CUDA FPS 后端。正式运行依赖仓库根目录下的 `adaptive_resampling_data_v3/`、`modelnet40_object80_seed21/` 与 `server_runs/` 下的原始权重和结果。路径不同可用 `--help` 查看覆盖参数。

```bash
(
  set -e
  cd /home/zk/baseline_compare
  git switch main
  git pull --ff-only origin main
  conda activate scdata-v3
  export CUDA_VISIBLE_DEVICES=0
  out=server_runs/our2_full_diagnosis_from60_seed21
  log=server_runs/our2_logs/full_diagnosis_from60_seed21.log
  mkdir -p "$out" server_runs/our2_logs
  nohup flock -n "$out/queue.lock" python -u -B tools/diagnose_our2_all.py \
    --output "$out" --whole-regression >>"$log" 2>&1 </dev/null &
  echo "PID=$!  日志=$log"
)
```

只做固定整物体示例时，删除 `--whole-regression`。正式开始前可执行 `python -u -B tools/diagnose_our2_all.py --debug 1`，每条件取一个 Patch，每训练组取一个 batch，产物写入 `debug_1/` 子目录。整物体回归和正式五轮的产物位于正式输出目录。

```bash
tail -f server_runs/our2_logs/full_diagnosis_from60_seed21.log
```

中断后用同一条启动命令续跑。队列逐任务验证已完成产物的 SHA256，训练使用各组 `last.pt` 从最近完整 epoch 恢复。若改动源码、运行时、输入或回归设置，使用新的 `--output`。同一个正式目录一次只运行一个进程，由 `flock` 防止重复启动。

收集整个 `server_runs/our2_full_diagnosis_from60_seed21/`，优先查看 `queue.json`、`failures.json`、`manifest.json` 和 `reports/diagnosis_summary.md`。汇总表 `comparison_all.csv`、`sample_diagnostics.csv`、`paired_summary.csv` 与各模块诊断表都在 `reports/`。日志最后的 `OUR2 DIAGNOSIS COMPLETED` 表示队列全部通过。出现 `PARTIAL` 时按 `failures.json` 查看失败和阻断项，其他已完成项仍可分析。
