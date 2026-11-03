# our1 自适应重采样

模型版本为 `ours1-rgdb-relflow-v1`。局部训练输入 256 点、输出 1024 点，整物体评测输入 2048 点、输出 8192 点。网络只接收退化输入。下面的命令在服务器仓库根目录、已激活的 `scdata-v3` 环境中执行。

## 结构与消融

| 实验 | 模型配置 | 初始化与 Flow |
| --- | --- | --- |
| A0 | 既有 PUFM | 中点初始化、原速度场，沿用已完成的对照 |
| A1 | `gdb` | RGR＋固定总数收益预算、原速度场 |
| A2 | `relation` | 中点初始化、中性预算和观测关系、条件速度场 |
| A3 | `gdb_flow` | RGR＋收益预算、含恢复候选和预算的关系条件速度场 |

GDB 保留修正后的输入锚点，再分配 M−N 个新增点。每区先保留最低配额，按截断的面积代理分配覆盖储备，其余配额通过 GainNet 成对交换。默认储备比例 `--coverage-fraction 0.5` 作用于扣除最低配额后的新增预算。训练教师用同一候选池和冻结生成器比较交换前后的 CD-L1，可交换点不足时记录实际交换量。新版重建教师缓存。

候选由不同局部邻点对插值得到，使用逐点可靠性加权，恢复查询参与部分候选的位移。关系模块在 X0 上构建一次 8 近邻图，根据观测可靠性、恢复来源和恢复置信度混合三类消息，把关系特征和区域预算投影成 FiLM 参数，在每次积分的速度预测头内调制特征。区域预算记录生成来源，实际覆盖按最终坐标重新计数。

## 分阶段训练

```bash
source env_v3.sh
export V3_DATA_ROOT="$V3_CODE_ROOT/adaptive_resampling_data_v3"
COMMON=server_runs/patch32/pufm_common_b8_w0_a8_fused/epoch_0060.pt
RUN=server_runs/patch32/ours1_gdb_flow

ours1 train --stage rgr --preset gdb --checkpoint "$COMMON" --batch-size 8 --epochs 20 --output "$RUN/rgr"
ours1 build-teacher --checkpoint "$RUN/rgr/last.pt" --samples 128 --pairs 4 --teacher-epochs 0 1 2 3 4 --output "$RUN/teacher.npz"
ours1 train-gain --checkpoint "$RUN/rgr/last.pt" --teacher "$RUN/teacher.npz" --epochs 60 --lr 1e-3 --output "$RUN/gain"
ours1 evaluate-gain --checkpoint "$RUN/gain/last.pt" --split validation --condition C2_hole_10 --samples 32 --pairs 8 --compare-allocations --output "$RUN/validation_gain.json"
```

先看验证集的 `paired_allocations`。每个 patch 在相同候选和 Flow 下对照均匀、密度、曲率和学习预算，同时记录初始化 CD、最终 CD、GT→输出距离和预算偏差。优先检查学习预算是否改善几何，并在正常和混合条件上复核覆盖。满足这个条件后进入联合阶段。

```bash
ours1 train --stage joint --preset gdb_flow --checkpoint "$RUN/gain/last.pt" --fm-endpoint uniform --geom-weight 1 --cover-weight 1 --retain-weight 0 --batch-size 1 --epochs 30 --lr 1e-5 --output "$RUN/joint"
```

这是 A 路线的起始配置，几何和覆盖分别对实际积分终点与原始 GT 计算两个方向的平方距离。用相同初始化再开一组 `--retain-weight 0.01`，另存输出，比较预算保持与几何质量。损失权重为实验起点。示例联合 batch 为 1，正式比较时统一或披露 batch、优化步数和耗时。A1 从同一 gain 权重用 `--preset gdb`，A2 从共同主干用 `--preset relation`。

若 A 路线显示均匀终点抵消预算，可在单独实验切换 `--fm-endpoint adaptive`。B 路线只在训练时从当前 V3 基础样本的干净曲面候选池取点，应用同一归一化和旋转，按区域精确取满配额，再做 OT 配对。任一区域真实参考点不足时，该样本回到原 GT。每轮日志报告 B 路线实际使用和请求的样本数。两条路线都用原始 GT 计算几何与覆盖损失。

## 诊断和正式评测

```bash
python -B our1/diagnose_legacy.py --root "$V3_DATA_ROOT" --baseline server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt --ours server_runs/patch32/ours_rgr_mgb_b8_w0_a8_fused_seed21/full/last.pt --output "$RUN/legacy_validation.json"
ours1 smoke --preset gdb_flow --smoke-object --output "$RUN/smoke.json"
```

固定验证集选模规则后，使用已完成联合阶段的非调试权重。两个数据集各 80 个整物体、7 个条件，正式成绩来自独立评测入口。

```bash
python -B run_subset_eval.py --phase check --models ours1 --ours1-checkpoint "$RUN/joint/last.pt" --root "$V3_DATA_ROOT"
python -B run_subset_eval.py --phase run --models ours1 --ours1-checkpoint "$RUN/joint/last.pt" --root "$V3_DATA_ROOT"
```

默认结果目录为 `server_runs/whole_object_eval80_ours1`。多个消融或终点实验使用各自的 `--output`。单模型 `ours1 predict` 另存每个样本的初始化、最终区域计数及预算偏差。旧版说明和 `adaptive_checks/` 记录的是复制时的旧实现。
