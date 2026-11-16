# PUFM 单模块几何分配独立实验

本目录实现 2026-11-15 的 D1 / D2 / D3 方案。三个入口分别加载各自目录下复制的 PUFM 网络源码，分别训练、存储权重。几何计算、预算、教师和训练协议在本目录共用，实现统一对照。

| 目录 | 实验性几何输入 |
| --- | --- |
| `D1/PUFM-main/` | 切平面循环角度的最大覆盖缺口 |
| `D2/PUFM-main/` | 12 / 24 邻居二次拟合在共同观测位置的高度差平方 |
| `D3/PUFM-main/` | 24 邻居中确定性选取 4 个有插值支持的留出点，重新拟合后的高度误差平方 |

`single_m2.py` 提供同构的局部编码和递减边际收益头。几何值按训练教师中的有效区域做中位数中心化及 P90-P10 缩放。PCA 退化、病态拟合和外推位置使用有效性掩码。公共 M2 输入为局部学习特征、中心位置、局部尺度、输入点占比及有效标志。

条件速度分支采用独立的 32 维公共局部特征、当前最近区域的预算占比、相对中心位移和时间，输出层零初始化。公共特征编码器归属生成器，S4 中随 PUFM 一起冻结。M2 的编码器由收益监督更新。这一接口选择让 S4 更新 M2 时共享教师对应的生成器保持固定。

所有版本使用 16 个区域 / 256 输入点，新增点预算 768，每区下限 4、上限 64，交换步长 8，最多 32 次交换。收益头输出以训练收益尺度归一化，交换阈值为该尺度的 0.01 倍。整物体推理按输入点数增加区域数。

候选池采用 32 个局部输入点的无序两两中点，合并重叠邻域中的重复点，按最近区域中心分配唯一归属，再做确定性 FPS 排序。区域中点少于配额下限时，从该中心到附近已观测点的线段细分补足，将有效且去重的补充点追加到中点之后。全局容量仍不足时，在相同区域上限内补充候选。每区配额受实际容量约束，补充后仍无法满足约束的输入会触发明确报错。初始点云保留输入锚点，Flow 输出允许全部点移动。

## 顺序训练

| 阶段 | 默认设置 | 输出目录 |
| --- | --- | --- |
| S1 | 冻结原始基线，分别比较原始中点、共同池均匀、半径密度代理、曲率、当前几何规则 | `screen/D1`、`screen/D2`、`screen/D3` |
| S2 | 共同预适配 10 轮，前 2 轮冻结主干，使用均匀 / 半径密度代理 / 曲率 / 随机合法预算 | `common_preadapt` |
| S3 | S2 的最佳生成器，固定抽取 4096 个 HDF5 训练行，每行 4 次交换 | `teacher` |
| S4 | 每个 M2 训练 20 轮，冻结共同 S2 生成器 | `D*/gain` |
| S5 | 每个生成器从共同 S2 权重出发，加载本版本 S4 最佳 M2，适配 10 轮，前 2 轮冻结主干 | `D*/adapt` |

S4 使用 SmoothL1 收益回归及权重 0.1 的收益方向损失，排除近零标签。教师误差为双向最近邻欧氏距离均值的平均，即项目 CD-L1。收益尺度取训练标签绝对值的 P90，方向噪声阈值取重复推理误差最大值的 5 倍与 1e-7 的较大值。S5 固定 M2，交替使用学习预算与四种合法预算训练。生成器首版只使用基于当前初始化重新匹配的 Flow Matching 损失。

轮数、学习率和教师规模属于此次选择的初始配置。S2 和 S5 遍历完整原始 HDF5，S4 遍历共享教师。M2 学习率 1e-3，条件分支 1e-4，解冻主干 1e-5。所有阶段 seed 21，默认 batch 8，Flow 推理 5 步。`--teacher-samples 0` 可使用全部训练行生成教师。

每轮使用固定 ModelNet40 100 个物体验证并按物体等权 canonical CD-L1 选模。S2 的验证初始化采用共同候选池均匀配额，S4、S5 使用本版本学习配额。每轮验证结束立即原子保存 `last.pt`，刷新最优时立即原子保存 `best.pt`，`train.jsonl` 记录本轮和最佳验证分数。`last.pt` 包含优化器和 RNG 状态。完成当前轮后保存续训状态，中断发生在轮内时会从该轮起点重跑。

S4 的 Spearman 和方向准确率记录为训练教师诊断。S5 后使用首个训练教师分片做收益漂移探测并写入 `teacher_drift.json`，其样本规模同时记录。真实效果及是否需要扩大教师、重建标签或调整预算，需结合正式验证结果判断。统一队列会完成三版训练，诊断结果保存在各输出目录。

## Git 上传

在本机 PowerShell 执行。

```powershell
Set-Location D:\dkwork\baseline_compare
git add -- geometry_m2 our/PUFM-main/baseline_validation.py
git diff --cached --stat -- geometry_m2 our/PUFM-main/baseline_validation.py
git commit --only -m "Add independent D1 D2 D3 PUFM geometry allocation experiments" -- geometry_m2 our/PUFM-main/baseline_validation.py
git push origin main
```

`baseline_validation.py` 的改动使续训时已经匹配的最佳权重文件保留原始文件内容，避免重复序列化改变文件哈希。仓库已有 `.pt/.npz/.log/server_runs` 忽略规则。

## 服务器统一命令

在服务器 Bash 执行。默认基线路径沿用交接文档，`run_all.py` 会在数据根目录递归寻找唯一匹配的原始训练 HDF5。存在多个同名文件时，用 `--pu1k-h5 /实际路径/文件.h5` 指定。

```bash
cd /home/zk/baseline_compare
git pull --ff-only origin main
conda activate scdata-v3
export CUDA_VISIBLE_DEVICES=0 V3_FPS_BACKEND=cuda
mkdir -p server_runs/geometry_m2_seed21
set -o pipefail

python -u -B geometry_m2/run_all.py \
  --root "$PWD/adaptive_resampling_data_v3" \
  --checkpoint "$PWD/server_runs/pufm_official_sparse_seed21/best.pt" \
  --output "$PWD/server_runs/geometry_m2_seed21" \
  --preadapt-epochs 10 --gain-epochs 20 --adapt-epochs 10 \
  --freeze-epochs 2 --teacher-samples 4096 --teacher-pairs 4 \
  --batch-size 8 --workers 4 --flow-steps 5 --seed 21 \
  2>&1 | tee -a server_runs/geometry_m2_seed21/queue.log
```

执行顺序为 S1 三版规则检查、共同 S2、共同 S3、D1 的 S4/S5、D2 的 S4/S5、D3 的 S4/S5。任何子进程失败，队列会返回失败。再次执行相同命令会自动从各阶段 `last.pt` 续训，已完成的训练阶段校验身份后返回，教师复用完整分片。可添加 `--skip-screen` 省略已经完成的 S1。代码、输入权重、训练数据或协议发生变化时，应使用新输出目录。

三个最终最佳权重分别为 `server_runs/geometry_m2_seed21/D1/adapt/best.pt`、`D2/adapt/best.pt`、`D3/adapt/best.pt`。冻结生成器的学习分配结果保存在各自 `gain/best.pt`。前台命令持续输出日志，后台运行可使用现有的 tmux 会话。

## 测试入口

以 D2 为例，在服务器设置 `root` 与 `run` 后执行。D1、D3 使用各自目录下的入口和权重。

```bash
root="$PWD/adaptive_resampling_data_v3"
run="$PWD/server_runs/geometry_m2_seed21/D2/adapt"

python -u -B geometry_m2/D2/PUFM-main/train.py predict \
  --root "$root" --checkpoint "$run/best.pt" --device cuda \
  --split modelnet40_holdout --modelnet40-objects-per-class 5 \
  --output "$run/modelnet40_200.npz"
python -u -B tools/evaluate_v3_predictions.py "$run/modelnet40_200.npz" \
  --root "$root" --split modelnet40_holdout --condition A0_normal \
  --modelnet40-objects-per-class 5 --output "$run/modelnet40_200_metrics"

export V3_RAW_ROOT="$root"
python -u -B geometry_m2/D2/PUFM-main/train.py predict \
  --root "$root" --checkpoint "$run/best.pt" --device cuda \
  --split official --output "$run/pu1k_official.npz"
python -u -B tools/evaluate_v3_predictions.py "$run/pu1k_official.npz" \
  --root "$root" --split official --condition official_A0 \
  --output "$run/pu1k_official_metrics"
```

PU1K 官方 XYZ 路径沿用原交接文档。全量 ModelNet40 2368 留出物体使用预测参数 `--modelnet40-objects-per-class 0`，评估时省略对应参数。测试 GT 仅由统一评估器读取，推理接口只接受输入点云。训练和教师只读取原始 PU1K 训练 HDF5，其原始 mesh 身份缺失的限制继续记录在产物中。

## 本机验证

```powershell
& D:/Anaconda/envs/pufm/python.exe -m unittest discover -s geometry_m2/tests -v
```

9 项回归检查覆盖几何有效性与 CPU/GPU 一致性、候选唯一归属、容量与整数预算、当前初始化的点集匹配、M2/条件分支/主干梯度、重复生成、配额改变对输出的影响、权重身份和最佳权重恢复。随机抽取的 256 个真实训练 patch 通过候选容量检查。

完整 GPU 短测使用预置 PUFM 权重、每轮 1 个训练 batch、1 个验证物体、8 个教师训练样本和 1 步 Flow，跑通三个版本及队列重复执行，结果用于检查程序流程。另已跑通规则筛选、留出集预测与统一评估入口。本机短测采用 PyTorch FPS 后端，服务器 CUDA FPS 下的完整训练及正式指标需在服务器验证。正式实验使用服务器上新训练完成的干净输入基线及完整 100 物体验证。
