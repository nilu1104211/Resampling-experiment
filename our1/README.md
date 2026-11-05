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

## 自动接续训练

在服务器激活 `scdata-v3` 后，可一次启动整个训练流程。程序先等待当前项目中已有的训练进程结束，再依次执行 RGR 20 轮、训练教师、GainNet 60 轮、三种验证条件的收益诊断、联合训练 30 轮。每阶段成功结束后自动进入下一阶段。诊断报告保存后自动继续联合训练，最终效果由完整验证和正式测试判断。

```bash
cd /home/zk/baseline_compare
conda activate scdata-v3
export RUN=/home/zk/baseline_compare/server_runs/patch32/ours1_gdb_flow
mkdir -p "$RUN"
nohup python -u -B our1/train_pipeline.py \
  --root /home/zk/baseline_compare/adaptive_resampling_data_v3 \
  --output "$RUN" >> "$RUN/pipeline.console.log" 2>&1 < /dev/null &
echo $! > "$RUN/pipeline.pid"
tail -n 40 "$RUN/pipeline.console.log"
```

主日志会实时合并各阶段输出，各阶段另有自己的日志。RGR、GainNet、联合训练分别写入 `rgr/console.log`、`gain/console.log`、`joint/console.log`。成功时主日志末尾打印 `PIPELINE COMPLETE` 和最终 best 路径，失败时打印 `PIPELINE FAILED`，后续阶段停止。

重新执行同一启动命令时，程序检查已有检查点。已完整训练并选出 best 的阶段会跳过，有有效中间 `last.pt` 的阶段从保存状态续训。已完成但尚未选择 best 的 RGR 会先补选。目录锁会阻止两个流水线同时写同一个运行目录。输入指纹用于检查父权重、数据和生成器代码是否在阶段之间发生变化。手动启动的 RGR 可以接入，手动启动的 GainNet 和联合训练需要使用各自已有命令管理，自动流程要求它们具有流水线输入记录。

默认从既有共同 PUFM 第 60 轮开始，下一阶段加载上一阶段的 `best.pt`。另起一组实验时，指定新的 `--output` 目录。

## 分阶段训练

```bash
source env_v3.sh
export V3_DATA_ROOT="$V3_CODE_ROOT/adaptive_resampling_data_v3"
COMMON=server_runs/patch32/pufm_common_b8_w0_a8_fused/epoch_0060.pt
RUN=server_runs/patch32/ours1_gdb_flow

ours1 train --stage rgr --preset gdb --checkpoint "$COMMON" --batch-size 8 --workers 0 --assignment-workers 8 --fused-optimizer --epochs 20 --output "$RUN/rgr"
ours1 build-teacher --checkpoint "$RUN/rgr/best.pt" --samples 128 --pairs 4 --teacher-epochs 0 1 2 3 4 --output "$RUN/teacher.npz"
ours1 train-gain --checkpoint "$RUN/rgr/best.pt" --teacher "$RUN/teacher.npz" --fused-optimizer --epochs 60 --lr 1e-3 --output "$RUN/gain"
ours1 evaluate-gain --checkpoint "$RUN/gain/best.pt" --split validation --condition C2_hole_10 --samples 32 --pairs 8 --compare-allocations --output "$RUN/validation_gain.json"
```

先看验证集的 `paired_allocations`。每个 patch 在相同候选和 Flow 下对照均匀、密度、曲率和学习预算，同时记录初始化 CD、最终 CD、GT→输出距离和预算偏差。优先检查学习预算是否改善几何，并在正常和混合条件上复核覆盖。满足这个条件后进入联合阶段。

```bash
ours1 train --stage joint --preset gdb_flow --checkpoint "$RUN/gain/best.pt" --fm-endpoint uniform --geom-weight 1 --cover-weight 1 --retain-weight 0 --batch-size 8 --workers 0 --assignment-workers 8 --fused-optimizer --epochs 30 --lr 1e-5 --output "$RUN/joint"
```

这是 A 路线的起始配置，几何和覆盖分别对实际积分终点与原始 GT 计算两个方向的平方距离。用相同初始化再开一组 `--retain-weight 0.01`，另存输出，比较预算保持与几何质量。损失权重为实验起点。RGR 和联合训练沿用旧配置的 batch=8，每轮 3188 次参数更新。GainNet 缓存训练继续使用独立的 `--gain-batch-size 32`。A1 从同一 gain 权重用 `--preset gdb`，A2 从共同主干用 `--preset relation`。

如果已有 batch=1 的联合训练检查点，自动脚本会提示配置不匹配。统一 batch=8 的实验应从 GainNet 的 best 权重启动新的联合训练目录。已有 RGR、教师和 GainNet 结果可以继续复用。

若 A 路线显示均匀终点抵消预算，可在单独实验切换 `--fm-endpoint adaptive`。B 路线只在训练时从当前 V3 基础样本的干净曲面候选池取点，应用同一归一化和旋转，按区域精确取满配额，再做 OT 配对。任一区域真实参考点不足时，该样本回到原 GT。每轮日志报告 B 路线实际使用和请求的样本数。两条路线都用原始 GT 计算几何与覆盖损失。

## 验证集最佳权重

`train` 的 baseline、rgr、joint 阶段，以及 `train-gain` 默认每 5 轮完整验证一次，最后一轮也执行验证。可用 `--val-every 1` 改成每轮验证，或用 `--val-every 0` 关闭。默认验证 100 个独立物体的全部 4 个 patch 和 7 个条件，共 2800 条记录。每次验证会增加一轮验证推理时间。

选模指标为共同 canonical 坐标下的 CD-L1。先在物体内平均 patch，再平均物体，然后在 A/B/C/D/E 五类内平均条件，最后五类等权平均。严格改善时更新，分数相同保留较早权重。RGR 阶段用修正锚点的中点初始化和冻结主干评价，GainNet、联合阶段用当前完整网络输出评价。

每个训练阶段输出以下文件。

- `best.pt` 保存验证指标最好的模型、对应优化器和随机状态，供下一阶段初始化或固定选模后测试。
- `last.pt` 保存最新训练状态，供同阶段 `--resume` 续训。
- `best.json` 记录所选轮次、分数、条件和数据指纹。
- `validation.jsonl` 保存每次验证的整体与分条件分数。

验证使用 eval 模式，结束后恢复模块模式和 Python、NumPy、CPU/CUDA 随机状态。续训继承之前的最佳分数与模型，复制运行目录时同步保留 `last.pt` 和 `best.pt`。训练阶段结束后，最佳权重写入该阶段已完成的总轮数，正式评测可据此接受较早轮次。

`--val-objects N` 可用少量物体做流程诊断。此类 `best.pt` 会记录非完整验证范围，正式评测要求完整验证选出的权重。

如果训练已经按旧命令启动，可以让当前阶段完成，再对保留的候选权重补选。以下命令在 RGR 20 轮完成后执行。

```bash
ours1 select-best --checkpoint "$RUN/rgr/last.pt" --output "$RUN/rgr"
```

它评价该目录的 `epoch_*.pt` 和末轮权重，生成 `best.pt`。若已经有新版本生成的 `best.pt`，直接使用该权重。下一阶段的教师构建和 GainNet 训练须共同加载同一个 RGR best，缓存指纹会检查生成器权重。

## 诊断和正式评测

```bash
python -B our1/diagnose_legacy.py --root "$V3_DATA_ROOT" --baseline server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt --ours server_runs/patch32/ours_rgr_mgb_b8_w0_a8_fused_seed21/full/last.pt --output "$RUN/legacy_validation.json"
ours1 smoke --preset gdb_flow --smoke-object --output "$RUN/smoke.json"
```

固定验证集选模规则后，使用已完成联合阶段的非调试权重。两个数据集各 80 个整物体、7 个条件，正式成绩来自独立评测入口。

```bash
python -B run_subset_eval.py --phase check --models ours1 --ours1-checkpoint "$RUN/joint/best.pt" --root "$V3_DATA_ROOT"
python -B run_subset_eval.py --phase run --models ours1 --ours1-checkpoint "$RUN/joint/best.pt" --root "$V3_DATA_ROOT"
```

默认结果目录为 `server_runs/whole_object_eval80_ours1`。多个消融或终点实验使用各自的 `--output`。单模型 `ours1 predict` 另存每个样本的初始化、最终区域计数及预算偏差。旧版说明和 `adaptive_checks/` 记录的是复制时的旧实现。

冻结 ModelNet40 测试包生成于 OUNet 的 `coordinates_only=False` 可选预检查路径加入之前。评测入口兼容这两个已核对的完整协议源码指纹，仍逐个校验选中物体的数据 SHA256、原始划分和条件；兼容依据及新旧协议指纹写入评测 manifest 的 `supplement_protocol`。未核对的协议变化继续报错，不需要修改或重新生成冻结数据。
