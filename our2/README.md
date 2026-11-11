# our2：PUFM 双独立模块

本目录实现 [DESIGN.md](DESIGN.md) 中的第一版 RAGR / GDBA。保留原 PUFM 速度场，新增代码只作用于初始化。新 M1/M2 与旧筛选 M1b/M2a 不是同一模块。

## 目录与来源

- `PUFM-main/models/diffusion.py`：直接复制 `baseline/PUFM-main`，未修改速度场。
- `pointnet2_util.py` / `utils.py`：复制原 PUFM，改为现有确定性 Torch 点运算接口；sklearn 仅在原辅助函数被调用时导入。
- `geometry.py` / `ragr.py` / `gdba.py` / `dual.py`：新几何分区、M1、M2 和模型封装。
- `run.py`：从已验证的第二批训练/评估工具适配，移除旧候选机制依赖；复用仓库 `adaptive_resampling` 的数据、精确匹配、点运算和指标，以及 `run_subset_eval.py` 的冻结整物体协议。
- `configs/`：8 个阶段配置；`pipeline.py`：阶段队列和匹配预算对照。
- `check.py` / `tests/` / `audit.py`：功能验收、边界测试和本地资料审计。
- `DESIGN.md` 是原始设计稿；实现细节、偏差及已验证范围以本文和 `VERIFICATION.md` 为准。

没有复制示例点云、图片、旧 notebook、CUDA 编译产物或预训练权重。现有源码快照没有附带独立许可证文件，沿用仓库既有第三方使用约束；上游出处及引用见 `baseline/PUFM-main/README.md`，作者 Zhi-Song Liu、Chenhang He、Lei Li，论文 *Efficient Point Clouds Upsampling via Flow Matching*。

## 已实现的机制与边界

M1 共享一个小型几何编码器，预测逐点可靠度与候选缺失需求。候选由可见邻域间隙和局部切平面产生；缺失监督仅在候选附近真实删除的 GT 表面上定义。可靠性用 v3 保存的位移/间距软标签，不将噪声标记简单等同于可靠度。以零初始化的 `tanh(gamma)` 和局部尺度限制种子位移；M1-only 不调用预算分配器。

M2 使用 FPS 锚点和互斥最近锚点归属，`K=ceil(n/32)`。独立分支只读取输入几何；联合分支可读取 M1 区域可靠度和缺失需求，训练时有条件 dropout。连续预算经过上下限约束和最大余数整数化，严格保持 `N=4n`。默认下限约均值的 0.5 倍、上限 2 倍，受实际候选容量约束。候选不够时用区域内凸插值补充，去掉重复坐标；退化到无法提供足够不同坐标时明确报错。

预算学习使用训练 GT 在原 Midpoint 下的区域覆盖误差和表面点数构造需求分布，以 KL 监督 logits。**硬整数配额与 FPS 不可导；没有将它们宣称为端到端可微采样。** `m2_soft` 保留原初始化学习需求；`m2_hard` 使用真实配额训练速度场。

当前 PUFM 没有独立、静态的 `encode(X)` 接口，特征随流状态和时间生成。第一版因此使用轻量几何分支，没有伪造设计稿中的 API，没有增加另一个完整骨干，也没有改变速度场、EMD 配对、5 步 Euler 时间表或最终 GT。原 PUFM 权重严格加载；双模块关闭与原始直接调用保持等价。

孔洞候选目前是保守的邻域切平面中点，并不等于完整拓扑补全。真实开口误补、跨薄壁连接、严重缺失召回、Patch→整云质量仍须正式实验检验。此版本没有引入 PUFM++ 时间表、第三个创新模块或熔池物理模型。

## 环境与必要权重

在仓库根目录运行。使用已有服务器 `scdata-v3` 环境即可；运行所需为 PyTorch、NumPy、SciPy、h5py、einops、tqdm。依赖沿用根目录 `requirements_v3.txt`。本机 Python 3.13 的 h5py 已安装在被忽略的 `tmp/module_screening_deps`；仅本机运行时设置该路径到 `PYTHONPATH`，不要复制依赖到本目录。

原始权重为 `server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt`，SHA256：

```text
c602029630079252fcd249078877ebaae1be083c99e829a82c8e15aafd01887e
```

数据和基线权重须单独传输，Git 不包含它们。服务器可设置：

```bash
export V3_DATA_ROOT=/home/zk/baseline_compare/adaptive_resampling_data_v3
export PUFM_BASELINE_CHECKPOINT=/home/zk/baseline_compare/server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt
```

## 验收与训练

先运行本地/服务器验收（不进行完整训练）：

```bash
python -B -m unittest discover -s our2/tests -v
python -B our2/check.py --root "$V3_DATA_ROOT" --baseline-checkpoint "$PUFM_BASELINE_CHECKPOINT" --output server_runs/our2_checks --whole-shape
```

单独续训对照：

```bash
python -B our2/run.py train --config our2/configs/baseline.json --output server_runs/our2_p0
```

默认恢复原始 Adam 一、二阶矩，速度场学习率降为 `1e-6`，模块学习率 `1e-4`，梯度范数上限 1；训练退化 epoch 从 30 开始。新阶段重建模块优化器；有已训练 flow 的直接前序阶段优先继承其 flow Adam 状态。`--resume` 恢复本阶段完整模型、优化器、RNG、epoch、历史和选模记录，要求源代码/配置/数据元信息/后端/输出路径一致。恢复时沿用原命令的父权重参数。

训练开始先验证 epoch 0，之后每轮验证。生成阶段按 7 条件等权、先物体内聚合的 CD-L1 选 `best.pt`；纯辅助头阶段按对应验证辅助损失选择，避免所有初始化相同导致永远选 epoch 0。每阶段仅保存 `best.pt` 和 `last.pt`，不保存每轮全量权重。`last.pt` 用于续跑，`best.pt` 用于阶段衔接和评估。`best.json` 明确所选轮次；可能选择 epoch 0，此时不能宣称续训有效。

正式阶段队列（需要先审查 P0；本次开发没有启动这个完整预算）：

```bash
python -B our2/pipeline.py --root "$V3_DATA_ROOT" --baseline-checkpoint "$PUFM_BASELINE_CHECKPOINT" --output server_runs/our2_seed21 --epochs 5 --seed 21
```

顺序为 M1 可靠性→缺失→种子引导、M2 软预算→硬预算、冻结 M1 的联合条件→联合微调，另有 3E/2E/7E 的续训对照。Joint 从**原始 flow 和两个独立模块头**开始，不混合两个独立微调后互不兼容的 flow 权重。其预算包含 M1/M2 前序训练成本。各对照共享 batch、flow LR、原 checkpoint 和训练样本/更新数量；冻结阶段参数数目和每步计算量不同，日志须一并报告时间、参数与峰值显存。

暂停后原命令加 `--resume`；队列用独占锁防止并发写入。强制断电留下的锁应先核对记录的 PID/主机，确认进程已退出后再手动移除。

调试流程可添加 `--epochs 1 --batch-size 1 --limit-batches 1 --limit-samples 1 --allow-debug`。这些权重均标记为 `partial_debug_run`，正式评估默认拒绝；每条件只取一个样本，仅检查流程。

## 验证、正式测试与消融

```bash
python -B our2/run.py validate --config server_runs/our2_seed21/configs/m1_refine.json --checkpoint server_runs/our2_seed21/m1_refine/best.pt --output server_runs/our2_seed21/eval_m1
python -B our2/run.py predict --config server_runs/our2_seed21/configs/joint_finetune.json --checkpoint server_runs/our2_seed21/joint_finetune/best.pt --split pu1k_object --output server_runs/our2_seed21/test_pu1k
```

ModelNet40 使用 `--split modelnet40_object`，补充数据根目录沿用 `run_subset_eval.py` 的 `--whole-data-root`。正式测试沿用冻结 80 物体×7 条件、2048→8192 协议，不作为调参入口。输出包含每样本输入哈希、CD-L1/CD-L2²、F-score（0.01）、HD、孔洞/稀疏覆盖、时间、参数量与预算诊断；不默认计算 P2F。预测仅收到 `input_xyz`，GT/掩码在损失与评分侧使用。

用 JSON 配置定义内部消融：`use_reliability`、`use_hole`、`joint_condition`、`budget_policy`（learned/uniform/density/curvature）、`minimum=0`、`train_flow`。固定规则对照需要进入 `m2_hard` 的 FM 适应阶段；其软阶段不应用训练器训练，因为它没有可学习预算目标。更改机制属于新实验，应使用新输出目录，不能拿不同配置直接重载旧检查点。固定总点数、评估对象、初始权重和更新预算。

最终至少 3 个 seed 分别运行新目录，按数据集、条件报告，辅以物体配对置信区间。现有完整训练结果不能由本次局部验收替代。

## Git 与成果保存

`our2` 仅放源码、配置、测试、设计和说明。训练/验证输出必须在本目录之外，默认使用被根 `.gitignore` 忽略的 `server_runs/our2`，本机验收在 `outputs/our2_local`。权重、数据、数组、日志和缓存均不进入 Git。检查后仅添加 `our2/` 相关文件；本次没有提交或推送，也没有改动原始 PUFM、our、our1 或历史实验。
