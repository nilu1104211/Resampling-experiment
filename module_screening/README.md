# PUFM 七个独立模块筛选实验

2026-11-05 已建立七份**物理复制、独立修改**的 PUFM。代码实现与接口检查完成；没有正式训练成绩，也未启动服务器任务。

## 目录和实际实现

| ID | 独立目录 | 默认变体 | 本轮接入 |
|---|---|---|---|
| M1 | `m1_endpoint_cd/PUFM-main` | `m1a` | 单步可微终点 + 平方 CD；`m1b` 单独启用初始化扰动 |
| M2 | `m2_apml/PUFM-main` | `m2a` | 与 M1a 相同终点预测，损失替换成参考仓库原版 PyTorch APML |
| M3 | `m3_gaussian_features/PUFM-main` | `m3a` | 观测输入 KNN 协方差、特征值比例、无向法向外积，经 MLP 并联到最后 128 维特征 |
| M4 | `m4_repkpoints/PUFM-main` | `m4a` | 参考 RepKPU 的 REM/RepKPoints；不接入 KP-Queries 或生成头 |
| M5 | `m5_adaptive_denoise/PUFM-main` | `m5a` | 自行适配的轻量去噪器 + 逐轮云级继续/停止判别；冻结 PUFM |
| M6 | `m6_grad_refine/PUFM-main` | `m6a` | 参考 Grad-PU 的 P2PNet 学习距离场，PUFM 输出上执行 1 步坐标梯度修正；`m6b` 为 3 步 |
| M7 | `m7_ats/PUFM-main` | `m7b` | 原 PUFM 权重不动，训练 patch 误差曲线反演出 5 步 ATS 时间表；`m7a` 为均匀网格 |

每个目录都有自己的 `config.json`、`screening.py`、`models/candidate.py`、`models/screening_model.py` 和原始 PUFM 源码。没有软链接、硬链接，也没有跨实验导入另一个实验的代码。`run.py` 只通过独立进程转发命令；公共 V3 数据读取、算子兼容和评价协议保留为只读依赖。

原始 `baseline`、`our`、`our1`、`compare-change` 和 `reference` 源文件未修改。原始脚本/示例保留用于对照，**本轮统一使用新增的 `screening.py` 入口**。没有复制原仓库预训练权重；七项统一读取下面固定的 V3 基线。

## 固定实验条件

- 原始代码提交：`5f5536ccbd9e31c919a8239c18378fa663e8f95c`。文件级来源在 `source_manifest.json`，含用户提供的六个参考代码快照哈希；参考目录没有 Git 元数据，未虚构来源 commit。
- 唯一初始化权重：`pufm_baseline_b8_w0_a8_fused/epoch_0030.pt`，SHA256 为 `c602029630079252fcd249078877ebaae1be083c99e829a82c8e15aafd01887e`；每份配置均固定此哈希。
- 保留原 attention PUFM、midpoint 初始化和 **5 次 Euler 速度评估**。M7 每次乘真实 `Δt`，五段之和为 1。
- 训练：V3 mixed、每物体 32 patch、256→1024、默认 batch=8；M1–M4 默认额外训练 20 轮、Adam、lr=1e-5。20 轮是待验证的起始预算，修改预算时必须匹配控制组。
- M1–M4 各自提供 `b_extra`：同一基线继续原 FM 训练。M3/M4 另提供 `mlp_control` 容量对照。所有副本支持 `b0` 关闭候选模块。
- M5/M6 只训练新增模块，冻结 PUFM 参数和 BatchNorm 缓冲。M5 使用训练集配对干净输入；M6 使用冻结 PUFM 在训练对象上的预测及 GT 距离监督。推理接口均只接受观测坐标。
- 每 5 轮及末轮在完整验证集上选 best，先按物体聚合，再对七条件等权 CD-L1 选最小值。`fscore_floor` 可设验证 F-score 门槛，默认 0 表示不额外约束。best 是验证候选中最优，不是逐轮全搜。
- 冻结测试复用原入口的 seed=21、80 物体选择及 ModelNet40 冻结包核验，严格 2048→8192。没有自动运行冻结测试。

## 服务器准备

在仓库根目录运行，继续使用已有 `scdata-v3` 环境，不编译参考仓库旧版扩展：

```bash
conda activate scdata-v3
cd /home/zk/baseline_compare
export V3_DATA_ROOT=/home/zk/baseline_compare/adaptive_resampling_data_v3
export PUFM_BASELINE_CHECKPOINT=/home/zk/baseline_compare/server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt
python -B module_screening/run.py m1 train --help
```

新增模块采用 PyTorch/NumPy/SciPy 和现有 V3 算子。训练数据读取还需要 `requirements_v3.txt` 中的 h5py/einops 等；本轮没有修改或降级服务器环境。

## 先检查接口

```bash
# 七个独立进程顺序运行；每项只读一个训练 patch，执行前向/反向/单步优化。
# 这是 debug 接口检查，不是短训筛选成绩。
python -B module_screening/check.py --real-patch --device cuda \
  --output outputs/module_screening_checks/server_smoke_01

# 不使用数据和权重的纯随机 toy 检查也支持 CPU。
python -B module_screening/run.py m5 smoke --device cpu
python -B -m unittest discover -s tests -p test_module_screening.py -v

# 仅检查整物体尺寸和五次速度调用，使用随机输入，不读取冻结测试集。
python -B module_screening/check_whole.py \
  --baseline-checkpoint "$PUFM_BASELINE_CHECKPOINT" \
  --output outputs/module_screening_checks/whole_shape_01
```

## 一键顺序运行

新增 `train_all.py`，单个子进程完成后才开始下一项，默认顺序：

`B-extra → M1a → M1b → M2a → M3a → M4a → M5a → M6a → M7校准 → M7均匀验证 → M7 ATS验证`。

训练默认 20 轮、batch=8、workers=0、assignment_workers=8；保留各模块原配置的学习率。B-extra 是 M1–M4 共享的等预算 FM 续训对照，仍从同一原 PUFM 权重开始。M7 不训练新网络。M1–M6 在各自训练内定期完整验证，队列不启动冻结测试。

完成本页“服务器准备”后，在服务器仓库根目录运行：

```bash
mkdir -p server_runs/patch32/module_screening
nohup python -u -B module_screening/train_all.py --epochs 20 --fused-optimizer \
  --output server_runs/patch32/module_screening/sequential_v1 \
  > server_runs/patch32/module_screening/sequential_v1.console.log 2>&1 &
echo "queue PID=$!"
tail -f server_runs/patch32/module_screening/sequential_v1.console.log
```

权重及验证产物在 `sequential_v1/runs/<任务名>/`，各次子任务日志在 `sequential_v1/logs/`，冻结配置在 `configs/`，总进度在 `queue.json`。任何任务失败均立即停止。确认原队列已退出后，用原选项增加 `--resume` 接续：

```bash
nohup python -u -B module_screening/train_all.py --epochs 20 --fused-optimizer --resume \
  --output server_runs/patch32/module_screening/sequential_v1 \
  >> server_runs/patch32/module_screening/sequential_v1.console.log 2>&1 &
```

恢复前核对原源码、数据、初始化权重和配置身份；已完成任务验证产物后跳过，未完成训练使用本阶段 `last.pt` 续训。若某阶段尚未保存 `last.pt` 就失败，保留其目录并检查错误，程序不自动删除或覆盖。完整性不足的校准/验证输出也会停下来供检查。队列锁防止重复启动同一输出目录。

其他用法：

```bash
# 仅打印计划，不加载权重、不创建文件。
python -B module_screening/train_all.py --dry-run --fused-optimizer

# 可选加入 M3/M4 容量对照；请使用新的输出目录。
python -B module_screening/train_all.py --fused-optimizer --with-capacity-controls \
  --output server_runs/patch32/module_screening/sequential_with_controls_v1

# 只运行指定任务；ATS 验证会自动加入其训练集校准任务。
python -B module_screening/train_all.py --tasks m1a m2a \
  --output server_runs/patch32/module_screening/endpoint_pair_v1

python -B -m unittest discover -s tests -p test_module_screening_queue.py -v
```

## 独立训练与控制组

以下是运行示例，本轮没有执行正式训练。输出目录必须是新目录，程序拒绝覆盖已有运行。

```bash
# M1a 与 B-extra：相同额外预算，分别从固定原始 PUFM 开始。
python -B module_screening/run.py m1 train --variant b_extra --epochs 20 --fused-optimizer \
  --output server_runs/patch32/module_screening/m1/b_extra_20
python -B module_screening/run.py m1 train --variant m1a --epochs 20 --fused-optimizer \
  --output server_runs/patch32/module_screening/m1/m1a_20
python -B module_screening/run.py m1 train --variant m1b --epochs 20 --fused-optimizer \
  --output server_runs/patch32/module_screening/m1/m1b_20

# M2a 与 M1a 只差终点损失。APML 保留上游 sum reduction，损失数值尺度不同。
python -B module_screening/run.py m2 train --epochs 20 --fused-optimizer \
  --output server_runs/patch32/module_screening/m2/m2a_20

# M3/M4 使用原 FM 损失；控制组各自独立输出。
python -B module_screening/run.py m3 train --output server_runs/patch32/module_screening/m3/m3a_20
python -B module_screening/run.py m3 train --variant b_extra --output server_runs/patch32/module_screening/m3/b_extra_20
python -B module_screening/run.py m3 train --variant mlp_control --output server_runs/patch32/module_screening/m3/mlp_control_20
python -B module_screening/run.py m4 train --output server_runs/patch32/module_screening/m4/m4a_20
python -B module_screening/run.py m4 train --variant b_extra --output server_runs/patch32/module_screening/m4/b_extra_20
python -B module_screening/run.py m4 train --variant mlp_control --output server_runs/patch32/module_screening/m4/mlp_control_20

# M5/M6 独立训练辅助网络，默认 lr=1e-4、20 轮；预算仍需验证选择。
python -B module_screening/run.py m5 train --output server_runs/patch32/module_screening/m5/m5a_20
python -B module_screening/run.py m6 train --output server_runs/patch32/module_screening/m6/m6a_20
```

若先做单 batch 生命周期排错，加 `--epochs 1 --batch-size 1 --limit-batches 1 --limit-samples 1` 并另选 debug 输出目录。该权重带 `partial_debug_run=true`，不能当正式权重使用。

每个训练目录保存 `config.json`、`manifest.json`、`train.jsonl`、`last.pt`、`best.pt`、`best.json` 和各次验证逐样本/逐物体指标。计时记录训练秒数、optimizer steps、见过的 patch、峰值显存、参数量及冻结参数范围。

```bash
# 续训严格保持原配置、初始化权重、数据及源码身份；不扩展原定总轮数。
python -B module_screening/run.py m1 train \
  --config server_runs/patch32/module_screening/m1/m1a_20/config.json \
  --resume server_runs/patch32/module_screening/m1/m1a_20/last.pt \
  --output server_runs/patch32/module_screening/m1/m1a_20
```

## M7 只校准时间表

```bash
# SHA256 排序选 256 个训练 patch，epoch=0；同一匹配路径遍历 51 个时间点。
python -B module_screening/run.py m7 calibrate \
  --output server_runs/patch32/module_screening/m7/calibration_256
python -B module_screening/run.py m7 validate \
  --schedule server_runs/patch32/module_screening/m7/calibration_256/schedule.json \
  --output server_runs/patch32/module_screening/m7/validation_ats
python -B module_screening/run.py m7 validate --variant m7a \
  --output server_runs/patch32/module_screening/m7/validation_uniform
```

时间表保存训练索引/sample ID、数据/基线/源码哈希、误差曲线和 CDF，不读取验证或测试 GT。`--limit-batches` 会把时间表标为 debug。不能使用 PUFM++ 的旧时间表代替本 PUFM 校准。

## 验证与冻结测试

```bash
# 读取训练时保存的配置，避免评测时隐式改变轮数/超参数身份。
python -B module_screening/run.py m1 validate \
  --config server_runs/patch32/module_screening/m1/m1a_20/config.json \
  --checkpoint server_runs/patch32/module_screening/m1/m1a_20/best.pt \
  --output server_runs/patch32/module_screening/m1/m1a_20_validation

# 仅在验证筛选完毕后执行。每个 split 独立保存，均是固定 80 物体×7条件。
python -B module_screening/run.py m1 predict --split pu1k_object \
  --config server_runs/patch32/module_screening/m1/m1a_20/config.json \
  --checkpoint server_runs/patch32/module_screening/m1/m1a_20/best.pt \
  --output server_runs/patch32/module_screening/m1/m1a_20_pu1k80
python -B module_screening/run.py m1 predict --split modelnet40_object \
  --config server_runs/patch32/module_screening/m1/m1a_20/config.json \
  --checkpoint server_runs/patch32/module_screening/m1/m1a_20/best.pt \
  --whole-data-root /home/zk/baseline_compare/modelnet40_object80_seed21 \
  --output server_runs/patch32/module_screening/m1/m1a_20_modelnet80
```

M2–M6 替换实验 ID、对应 config 和 checkpoint 即可；M7 提供 `--schedule` 而非新 checkpoint。M6 可在同一训练配置和同一 checkpoint 上用 `--refine-iterations 1` / `--refine-iterations 3` 分别做 M6a/M6b 推理消融，输出必须分目录，实际步数写入 summary 和逐样本诊断。

输出包括 canonical 坐标预测、`per_sample.csv`、`per_object.csv`、七条件及总平均 `summary.json`；附去噪/精修位移诊断、冻结选择计划及哈希。区域指标仅在对应标签非空时存在，`metric_condition_counts` 记录参与条件数；核心 CD/HD/F-score 始终按七条件平均。计时包括辅助模块，未统一预热，不能直接写成严格加速比。

## 检查结果与适用范围

本机 QA 使用 Python 3.13.3、PyTorch 2.7.1+cu118、实际 V3 训练 patch 和固定基线，结果保存在 `outputs/module_screening_checks`。这不等于已在服务器 Python 3.10/PyTorch 2.5.1+cu121 上验证，也不表示模块已改善指标。

已通过 11 项隔离/数值/求解器单元检查、七分支真实 patch 前向/反向/单步优化检查（M7 无优化）、七分支随机整物体 2048→8192 检查，以及 M1 训练→验证→best/last 保存/读取、完成阶段续训入口与 M7 校准→验证生命周期。详见 `VERIFICATION.md`。

M3a 是方案中的协方差版本；M3b 可学习高斯编码和 M5b 联合训练属于后续变体，本轮没有实现。M4 保留原 FM 目标，没有叠加 REM 正则损失；核点半径沿用参考默认值，整物体迁移仍需验证。M5 使用云级继续/停止监督，是 ASDN 思想适配，不能标成官方 ASDN 复现。M6 包含输出 Softplus 和步长截断适配。详细来源与改动见 `THIRD_PARTY.md`。
