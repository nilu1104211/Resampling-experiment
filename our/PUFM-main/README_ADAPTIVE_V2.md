# PUFM + RGR-v2 + MGB-v2：V3 实现说明

Linux 服务器使用新增 `run_adaptive.sh`，正式入口自动定位项目；数据位置支持 `V3_DATA_ROOT` / `--root`，原始文件映射支持 `V3_RAW_ROOT`。完整命令见 [服务器实验运行说明](../../服务器实验运行说明.md)。

本实现位于 `our/PUFM-main`，以用户的 v2 方案为设计依据。保留原 PUFM/PUFM_w_attn 的网络层与权重形状，新增可靠几何恢复、反事实收益预算、候选生成、条件速度残差及 V3 分阶段入口。原 `train_pufm.py/test_pufm.py` 留存追溯，正式使用新增的 `adaptive_v3.py`。

这是实现与短流程验收，尚未完成正式长训练、性能证明或系统性查新。随机初始化和短训练的权重均标记 debug，不能作为论文结果。

## 数据与接口

- 直接使用已有 V3：Patch `256→1024`、整物体 `2048→8192`；不重新生成输入、GT 或退化。
- 所有模型几何输入来自实际 `input_xyz`；原始近似面积均匀 GT 保持不变，只用于训练损失和评价。
- 模型接口：`model(points[B,N,3], output_size=M)`；不接受 GT、干净输入、孔洞标签等推理参数。
- V3 的标签只传给 `adaptive_losses.py` 或 `gain_teacher.py`，不送进模型前向。
- 归一化、旋转及逆变换复用 V3；全物体采用 whole 积分，预算初始化后没有全局 FPS 合并。
- 训练默认 `PUFM_w_attn`，可在 AdaptiveConfig 中显式选择 `pufm`，不能混用权重。
- 保留的原主干第一层采样 1024 点，因此查询点数 M 至少为 1024；其他 M 还需满足区域容量约束。

## 模块与实际实现

| 文件 | 内容 |
|---|---|
| `models/rgr_v2.py` | k=8/24 共享多尺度 Edge-MLP 编码；q、限幅 ΔX；区域 h/r/u 与偏移恢复查询 V |
| `models/marginal_gain.py` | 小 MLP 预测正的 a/β/γ，表示递减单点边际收益；Δ 点收益严格按连续 Δ 项求和；static 形式作对照 |
| `models/budget_allocator.py` | 输入锚点唯一预算归属、覆盖初始化、整数配额、供体损失与受体收益的成对交换、嵌套候选池 |
| `models/adaptive_wrapper.py` | RGR→区域预算→X0→原 PUFM；可选条件速度残差，逐积分步重新查询实际锚点 |
| `models/adaptive_losses.py` | 原 GT 的 FM 速度监督、RGR 辅助监督、独立 GainNet 教师损失、默认关闭的弱预算保持 |
| `gain_teacher.py` | 只允许训练集构建缓存；同候选池与生成器的固定总数交换；记录输入/GT/生成器哈希、索引和 epoch |
| `adaptive_v3.py` | baseline/rgr/gain/joint 分阶段训练、续训、预测、留出验证收益诊断、短检查 |

默认每 256 个输入点分 32 个区域；每区新增点下界 4、容量 64。Patch 保留 256 个修正锚点，新增预算 768。整物体按输入规模扩为 256 区，新增预算 6144。严重离群点首版仍保留为受限修正锚点，暂未实现锚点删除与替换策略。

初始化候选是局部中点池，可靠性缩放局部位移，h/r/u 控制恢复查询偏移；普通候选和恢复候选交错排列。每区先建立一次固定顺序的候选池，再按预算取前缀。改变配额不会重新随机生成所有区域的候选，也不会最后全局 FPS 抹掉配额。

预算归属代表生成来源；候选的几何位置可能越过最近区域分界。诊断同时记录新增预算、锚点计数、初始化实际空间计数和最终空间计数，不能把来源标签当成几何区域计数。

### 监督的真实含义

- q：监督来自 V3 `reliability_target_input`，它是合成可靠性代理，不是真实测量置信度。
- ΔX：来自逐点对应的 `denoise_offset`，位移限幅为局部尺度的 2 倍。
- h：由 GT 孔洞标记在当前区域内的比例构成软目标，不把低输入密度直接标成孔洞。
- V：训练时将区域缺失 GT 的中心投影到实际缺失 GT 点，提供恢复查询目标；推理不进行这一步。
- r：由缺失目标附近的可见干净输入支持距离构成可恢复性代理，没有宣称数据已有真实 r 标签。
- u：由预测查询到干净 GT 样点的相对距离构成停止梯度的几何风险代理，不是严格曲面距离或已经校准的贝叶斯不确定性。

默认完整模型保留这些分支，并提供 `--no-reliability`、`--no-recovery`、`--no-uncertainty`、`--no-virtual`，无效果的分支应在最终实验中删减。

## 反事实收益教师

构建基准预算 m，生成 Y(m)；仅向 A 移入 Δ 点、从 B 移走 Δ 点，生成 Y(m')。标签是共同 canonical 坐标中的原 GT CD-L1 改善：

`improvement = CD(Y(m), G) - CD(Y(m'), G)`。

同一记录的模型权重、候选池、积分步数与其他区域配额保持一致。起始预算在 uniform/density/curvature 间变化，Δ 在默认 4/8/16 间变化，避免只在一个配额上拟合。候选生成无随机扰动；每个基准只积分一次后复用于多个交换。

GainNet 拟合的是 `sum(g_A(m_A+i))-sum(g_B(m_B-Δ+i))`，不是直接回归 GT 曲率。采用 Huber 与轻量符号排序监督，标签放大 1000 倍只为数值尺度；实际收益诊断再还原。

缓存保存生成器/RGR/条件分支与配置指纹、点算子后端、样本 ID、索引、epoch、实际输入哈希及 GT 哈希。更改这些生成因素后，旧缓存会被拒绝。训练 GainNet 本身不改变指纹。

联合训练中，生成器会变化，因此每 10 个 batch 默认用当前实际生成器做一次在线交换监督，继续更新收益头，避免只拿旧冻结生成器的标签长期联合训练。此额外训练开销必须报告。GT 不参与最终推理。

硬配额、取整、FPS 不提供 GainNet 的有效几何梯度。本实现明确用 L_gain 训练预算头，没有将其称为完整端到端可微模型。

## 运行

本机默认使用已有 `D:/Anaconda/envs/pufm/python.exe`（Python 3.9、PyTorch 2.5.1、CUDA）。用 `ADAPTIVE_PYTHON` 可覆盖解释器，迁移数据代码时设置 `V3_CODE_ROOT`。数据目录通过 `--root` 覆盖。

优先调用已安装的原 PUFM `pointops_cuda`；CPU 或未安装扩展时，使用纯 PyTorch FPS/KNN/ball-query，不触发运行时 CUDA 编译。切换算子后端会改变教师指纹，需要重建缓存；后端与原算子的选点顺序可能不同，须记录实验环境。

EMD 对齐采用与对比网络适配相同的 SciPy 精确 Hungarian 平方距离匹配，替代原近似 CUDA EMD。FM 速度目标和原始 GT 保留，但损失按元素均值归约，避免 batch/点数改变时尺度失控。内部消融统一使用该实现；不要把 V3 数值直接称为原论文官方复现。

```powershell
# 在 our/PUFM-main 中运行。下面轮数是配置示例，不是已验证最优值。
.\run_adaptive.ps1 train --stage baseline --preset baseline --epochs 60 --workers 2 --output adaptive_runs/baseline

# 从 V3 基线权重开始训练 RGR，主干冻结。
.\run_adaptive.ps1 train --stage rgr --preset full --checkpoint adaptive_runs/baseline/last.pt --epochs 20 --workers 2 --output adaptive_runs/rgr

# 128 个基础 Patch × 5 个在线 epoch × 每例 4 次交换；只用训练集。
.\run_adaptive.ps1 build-teacher --checkpoint adaptive_runs/rgr/last.pt --samples 128 --pairs 4 --teacher-epochs 0 1 2 3 4 --output adaptive_runs/teacher.npz

.\run_adaptive.ps1 train-gain --checkpoint adaptive_runs/rgr/last.pt --teacher adaptive_runs/teacher.npz --epochs 60 --lr 1e-3 --output adaptive_runs/gain

# 先保留原 FM，不预设必须增加条件或保持损失。
.\run_adaptive.ps1 train --stage joint --preset full --checkpoint adaptive_runs/gain/last.pt --epochs 30 --lr 1e-5 --workers 2 --output adaptive_runs/full

# 若需要验证条件 FM，则单独使用 full_condition；retain 默认 0。
.\run_adaptive.ps1 train --stage joint --preset full_condition --checkpoint adaptive_runs/gain/last.pt --retain-weight 0.01 --epochs 30 --lr 1e-5 --output adaptive_runs/conditioned

.\run_adaptive.ps1 predict --checkpoint adaptive_runs/full/last.pt --split pu1k_patch --condition C2_hole_10 --output adaptive_predictions/full_C2.npz
.\run_adaptive.ps1 predict --checkpoint adaptive_runs/full/last.pt --split pu1k_object --condition E2_mixed_outliers --output adaptive_predictions/full_object_E2.npz

# 共同评价器：ID 配对、原 GT、canonical 逆变换、物体级统计。
& D:/Anaconda/envs/pufm/python.exe -B eval_adaptive.py adaptive_predictions/full_C2.npz --split pu1k_patch --condition C2_hole_10 --coordinates model --p2f --output adaptive_evaluation/full_C2

# 留出验证：只评估交换预测，不用验证数据更新 GainNet。
.\run_adaptive.ps1 evaluate-gain --checkpoint adaptive_runs/gain/last.pt --split validation --condition C2_hole_10 --samples 32 --pairs 8 --output adaptive_evaluation/gain_C2.json
```

`train` 与 `train-gain` 均保存可恢复的 `last.pt`，以及每 5 epoch 一个候选权重；分别有 `train.jsonl/gain.jsonl`。通过 `--resume last.pt --epochs 总轮数` 续训，保留配置/阶段及优化器。权重选择使用完整独立验证集；当前不自动宣称 last 等于 best，先验证候选权重再固定测试。

联合阶段 `retain_weight=0` 为默认。弱预算保持全展开积分，增加训练显存；它可能与跨区域虚拟查询恢复相互制约，是否保留由几何质量与覆盖决定，不能只看预算相似。

## 消融与检查

已有 preset：baseline、rgr、rule、static、mgb、rgr_static、full、full_condition。static 与 MGB 使用相同结构的 trunk 和同一训练期反事实缓存；static 取消 m 依赖，按普通 Softmax 配额分配。相同 RGR 状态下可以共用缓存，必须从同一冻结生成器训练两个预算头。

```powershell
.\run_adaptive.ps1 train-gain --preset rgr_static --checkpoint adaptive_runs/rgr/last.pt --teacher adaptive_runs/teacher.npz --epochs 60 --output adaptive_runs/rgr_static
.\run_adaptive.ps1 smoke --preset full_condition --smoke-object --output adaptive_checks/smoke.json
& D:/Anaconda/envs/pufm/python.exe -B -m unittest discover -s tests -p test_adaptive_contract.py -v
```

`--max-batches`、`--allow-debug`、预测 `--limit` 只用于调试。调试权重和教师都标记 debug，正式预测和教师默认拒绝使用这类权重；显式放行仍保留 debug 标记。

每个预测另存 `.diagnostics.jsonl`：预算、初始化/最终点数、区域中心、q/ΔX/h/r/u/V 及模型耗时。耗时包括 RGR、区域划分、候选与预算、全部 FM 积分；不包括事后 GT 评价和诊断日志计算。密度变化是机制诊断，不能单独证明性能。

工程验收见 `adaptive_checks/acceptance.json`；创新边界与需要证明的命题见 `NOVELTY_REVIEW.md`。
