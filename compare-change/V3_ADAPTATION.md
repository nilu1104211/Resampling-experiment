# 三个对比网络的 V3 适配

Linux 服务器使用新增的 `run_v3.sh`；路径迁移、分段保存候选与完整实验命令见 [服务器实验运行说明](../服务器实验运行说明.md)。正式入口现已按项目位置定位代码，数据位置支持 `V3_DATA_ROOT` / `--root`；原始 XYZ 与 P2F 网格支持 `V3_RAW_ROOT`，不需要改写原数据中的 Windows 路径。下文的 `.ps1` 和本机环境路径属于 Windows 使用说明。

本目录是 `compare` 的适配副本。正式 V3 入口为各网络下新增的 `v3.py`；原始训练、评测脚本仍保留用于追溯，不能把旧入口当作已经完成 V3 接入。

统一协议：训练 Patch `256→1024`，整物体 `2048→8192`，倍率均为 4。直接复用 `adaptive_resampling.v3_dataset`，没有重做、插值或重新抽取 V3 输入和 GT。训练采用物体级 V3 划分，`mixed/clean` 可切换；在线退化通过 EpochSampler 将 epoch 传给 worker。验证与测试读取冻结样本。

模型的几何输入只有实际 `input_xyz`；GT 只进入各模型本来需要的训练损失。所有预测保存到 V3 model 坐标系，之后由公共评价器恢复共同 canonical 坐标。不会对预测单独归一化后评价。

## 已调整的内容

| 方法 | 保留的结构与目标 | V3 适配 |
|---|---|---|
| PUFM++ | 原 RIN：6 个 RCW block、每 block 4 层 compute、256 个 latent、z_dim=512、x_dim=512；s1 FM 与 s2 Chamfer 微调 | 初始化只来自实际退化输入；修复 GlobalNet 的 FPS 接口和特征索引维度；训练使用 1024 个 query；整物体使用 256 点观测 Patch 分别生成并还原合并；s3 在训练集校准时间调度 |
| SPU-IMR | 原三次迭代 Mask-recovery、Transformer、原自监督恢复目标 | 主对比 `shared-v3` 严格读取相同 256 点输入，64 个中心×16 点，保留 8 次恢复和密度过滤，不在过滤前去重；最终 FPS 为 1024 点，整物体合并为 8192 点；保留源 EMD 与 cosine，历史 native-1024 仅用于独立推理参考 |
| OUNet | 原八叉树 CNN，逐层占用分类与叶节点位移回归 | 用固定 0.5 缩放替代原乘 1.7 后裁剪；输入与 GT 使用同一变换，越界报错而不裁点；推理按占用得分选叶节点，最终恰好 M 个，无复制点补齐 |

**训练目标的区别必须在论文中披露：** SPU-IMR 保持自监督，其恢复目标来自实际输入内部的被遮挡邻域，不使用 V3 干净 GT 计算损失。因此 mixed 训练中的噪声也可能被其重建；这是原目标在该任务上的适配结果，不能称为与另两种方法相同的干净 GT 监督。所有方法共享数据划分、实际观测、退化协议和评价。

**传输算子：** PUFM++ 当前对齐仍使用 SciPy Hungarian 平方距离匹配，不能称为原 CUDA EMD 的完全等价实现。SPU 的 `shared-v3` 默认恢复源近似 EMD 欧氏代价，缺少旧扩展时通过 NVRTC 运行保留的源 CUDA 核函数。SPU 的 Hungarian 仅作为显式后端变体或旧权重协议保留。

**OUNet 固定预算推理：** 原来输出点数随占用叶节点变化。本适配在每层以占用 logit 差排序，至少保留 `ceil(M/8^(D-d))` 个祖先节点，并将超额探索限制为该数量的 `--octree-beam` 倍（默认 4）；最后一层选恰好 M 个叶节点。训练仍使用原 GT 八叉树和原损失。这是显式推理适配，应标为 OUNet-V3 fixed-budget，记录原生官方协议结果时另列。由于固定 M，网络被迫输出足够数量的叶节点，效果需训练后评估。

PUFM++ 保留基于真实输入的回投约束（默认步长 0.01），不裁剪预测坐标。正式推理需提供训练集校准的 `--schedule`；显式指定 `--uniform-schedule` 才允许均匀时间的消融。`--manifold-step 0` 用于关闭回投的消融。Patch 使用观测计算一次归一化，整个积分过程使用同一坐标系。

## 运行环境

已在本机 `D:/Anaconda/envs/dianyun/python.exe`（Python 3.10、torch 2.5.1+cu121）验证。新增依赖安装在独立目录 `D:/dkwork/scdata/network_adaptation/deps`，没有修改现有 Conda 包。

目录中的 `run_v3.ps1` 自动使用上述解释器和依赖路径。可用 `V3_PYTHON`、`V3_DEPENDENCIES` 覆盖；数据代码迁移后设置 `V3_CODE_ROOT`，数据位置用 `--root`。新机器按 `requirements_v3.txt` 安装；PyTorch 与 torchvision 需匹配。

以下命令在本目录运行，`pufmpp/spu-imr/ounet` 分别对应三个网络。

```powershell
.\run_v3.ps1 pufmpp check-data
.\run_v3.ps1 spu-imr check-data
.\run_v3.ps1 ounet check-data

# 短检查：真实 V3 样本、前向、反向、优化一步、精确输出点数。
# 随机权重检查不代表训练完成或具备重建质量。
.\run_v3.ps1 pufmpp smoke --smoke-object --output v3_checks/pufmpp.json
.\run_v3.ps1 spu-imr smoke --smoke-object --output v3_checks/spu.json
.\run_v3.ps1 ounet smoke --smoke-object --output v3_checks/ounet.json
```

## 正式训练与推理命令

以下轮数是可运行配置示例，不表示已验证收敛或最优；默认未启动正式长训练。模型各自的训练成本、轮数与验证选模需完整报告。

```powershell
# PUFM++ 第一阶段
.\run_v3.ps1 pufmpp train --stage s1 --epochs 60 --batch-size 1 --workers 2 --training-mode mixed --output v3_runs/pufmpp_s1
# 第二阶段从第一阶段权重开始，重新建立优化器
.\run_v3.ps1 pufmpp train --stage s2 --checkpoint v3_runs/pufmpp_s1/last.pt --epochs 60 --lr 2e-5 --workers 2 --output v3_runs/pufmpp_s2
# 第三阶段只在训练集校准，默认遍历完整训练集和 51 个时间位置
.\run_v3.ps1 pufmpp schedule --checkpoint v3_runs/pufmpp_s2/last.pt --flow-steps 6 --output v3_runs/pufmpp_schedule.pt

.\run_v3.ps1 spu-imr train --spu-profile shared-v3 --spu-transport source --epochs 60 --spu-schedule-epochs 60 --spu-warmup-epochs 5 --batch-size 8 --workers 0 --training-mode mixed --output v3_runs/spu_shared256
.\run_v3.ps1 ounet train --epochs 60 --batch-size 2 --workers 2 --training-mode mixed --output v3_runs/ounet

# 续训使用 --resume，--epochs 表示总轮数。
.\run_v3.ps1 spu-imr train --spu-profile shared-v3 --spu-transport source --resume v3_runs/spu_shared256/last.pt --epochs 60 --spu-schedule-epochs 60 --spu-warmup-epochs 5 --batch-size 8 --workers 0 --output v3_runs/spu_shared256

# 固定条件逐项推理；完整条件名称见 V3 使用说明。
.\run_v3.ps1 pufmpp predict --checkpoint v3_runs/pufmpp_s2/last.pt --schedule v3_runs/pufmpp_schedule.pt --split pu1k_patch --condition C2_hole_10 --output predictions/pufmpp_C2.npz
.\run_v3.ps1 spu-imr predict --checkpoint v3_runs/spu/last.pt --split pu1k_patch --condition C2_hole_10 --output predictions/spu_C2.npz
.\run_v3.ps1 ounet predict --checkpoint v3_runs/ounet/last.pt --split pu1k_patch --condition C2_hole_10 --output predictions/ounet_C2.npz

# 改成 --split pu1k_object 时输出自动切换为 8192 点。
# --split modelnet40_patch 用于跨域测试；--split official 读取官方原始点对。
```

训练入口默认只保存可恢复的 `last.pt` 与 `train.jsonl`。`--validation-samples N` 可做验证监控；它不是物体级正式汇总，也不自动选出 best checkpoint。正式比较时应使用完整独立验证集选择 checkpoint，固定后再测试。可通过 `--epochs` 与 `--resume` 分段训练、保存候选 checkpoint。

`--max-batches`、`--limit`、`--schedule-batches` 只用于调试，输出记录 `partial_debug_run`；校准出的调试 schedule 会被正式推理拒绝。`--spu-groups`、`--octree-scale`、`--octree-depth` 与对应权重必须一致。若新输入范围超出八叉树域，应统一调小整次实验的固定 scale 后重训，不能对每个预测单独改尺度。

## 公共评价

预测 NPZ 包含 Unicode `sample_id[S]` 与 `prediction_xyz[S,M,3]`，另存 JSON 记录参数、协议、耗时与后处理。每个条件使用相同样本顺序和 ID；不读取干净输入或 GT 决定生成位置。

```powershell
$env:PYTHONPATH='D:/dkwork/scdata/network_adaptation/deps'
& D:/Anaconda/envs/dianyun/python.exe -B D:/dkwork/scdata/tools/evaluate_v3_predictions.py predictions/spu_C2.npz --split pu1k_patch --condition C2_hole_10 --output evaluation/spu_C2 --coordinates model --p2f
```

评价器做 canonical 逆变换、精确点数检查、区域指标与物体级统计。小样本调试文件需显式 `--allow-subset`。官方原始点对的预测可读取后用公共 `evaluate_sample` 评价；现有批量评价 CLI 面向 V3 自制固定条件 split。

本轮执行的短检查及限制见 `v3_checks/acceptance.json`。正式性能、收敛速度、长训练稳定性仍需后续实验确认。
