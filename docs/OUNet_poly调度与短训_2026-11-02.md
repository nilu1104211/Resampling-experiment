# OUNet poly 调度与短训

日期：2026-11-02，北京时间。本次在 Windows 本机仓库修改和验证代码；服务器需同步本次代码后再执行新增命令。

## 已确认的服务器环境

用户提供的服务器输出已确认：`scdata-v3` 中 PyTorch `2.5.1+cu121`、ocnn `2.3.3`、RTX 3080 Ti 可用。CUDA smoke 的随机权重前向、反向、一步更新、256→1024 和整物体 8192 点输出通过。该记录不能代替 batch=8 连续训练检查，也不代表训练质量。

## 新的训练行为

- OUNet 训练使用 AdamW，默认 `lr=5e-4`、`weight_decay=0.05`。
- poly 公式：`lr(e) = lr0 * max(0, 1 - e / H)^p`；`e` 为已经完成的 epoch，`H` 为固定 horizon，`p` 默认为 `0.9`。无 warmup。
- 第一轮使用 `lr0`；每轮所有 optimizer 更新完成后调用一次 scheduler，保存下一轮 LR。最后一轮训练的 LR 仍为正，完成 `H` 轮后保存 LR 为零。
- `--ounet-schedule-epochs H` 默认取 `--epochs`。分段训练时必须从第一段起显式传入完整 `H`，续训保持不变；`--epochs` 是这一次运行到的累计轮数，必须不超过 `H`。
- `last.pt` 保存 optimizer、scheduler、轮数、参数及 RNG。`--resume` 校验 horizon、power、初始 LR、weight decay、batch size、八叉树深度/尺度及公共训练数据选择；调度进度必须与已完成轮数一致。
- 旧的固定 LR checkpoint 缺少新调度状态，不能无声当作 poly 续训；如要复用其权重，显式用 `--checkpoint` 新开阶段和输出目录，优化器与调度重新开始。
- 调试 checkpoint 带 `partial_debug_run=true`，不能直接 `--resume` 为正式完整训练。正式训练应从头建立独立目录，不能沿用下面的调试权重。
- 控制台打印调度类型、配置和 LR；`train.jsonl` 记录本轮 `lr` 和保存后的 `next_lr`。
- SPU 保留原有 cosine 调用时机及两项恢复校验，只把检查键移到它自己的 Adapter 中。

公式和默认指数核对自原框架 [thsolver poly 实现](https://github.com/octree-nn/solver-pytorch/blob/master/thsolver/lr_scheduler.py) 与 [默认 lr_power 配置](https://github.com/octree-nn/solver-pytorch/blob/master/thsolver/config.py)。这里仍为 OUNet-V3 fixed-budget 适配；共同训练输入 256 点、GT 1024 点、每对象 32 个 patch 和原损失均保持一致。

## 服务器连续短训

先通过 Git 同步本次修改，再在服务器执行。这里的 `H=60` 仅用于短检查，不是确定正式长训轮数。若目录已有旧调试输出，换一个新的目录名。

```bash
conda activate scdata-v3
cd ~/baseline_compare
source env_v3.sh

v3 ounet train \
  --device cuda --seed 21 \
  --batch-size 8 --workers 0 \
  --training-mode mixed --train-patches-per-object 32 \
  --epochs 2 --ounet-schedule-epochs 60 --ounet-poly-power 0.9 \
  --lr 5e-4 --weight-decay 0.05 \
  --octree-depth 8 --octree-scale 0.5 \
  --max-batches 20 --log-every 5 \
  --output server_runs/debug/ounet_poly_b8_20261102
```

应看到 `scheduler: OUNetPolyLR`，第一轮 `lr=0.0005`，第二轮 `lr≈0.0004924937115`，第二轮结束 `next_lr≈0.0004849746890`，两轮均为 `batches=20`、`partial_debug_run=true`。若 CUDA 显存不足，保留完整错误，并在新目录中把 batch 降为 4 重新检查。

可再验证一次调试恢复，其他训练参数保持一致：

```bash
v3 ounet train \
  --device cuda --seed 21 \
  --batch-size 8 --workers 0 \
  --training-mode mixed --train-patches-per-object 32 \
  --epochs 3 --ounet-schedule-epochs 60 --ounet-poly-power 0.9 \
  --lr 5e-4 --weight-decay 0.05 \
  --octree-depth 8 --octree-scale 0.5 \
  --max-batches 20 --log-every 5 \
  --resume server_runs/debug/ounet_poly_b8_20261102/last.pt \
  --output server_runs/debug/ounet_poly_b8_20261102
```

恢复日志应显示 `start_epoch_zero_based=2`、`lr≈0.0004849746890`，只增加第三轮。这只能继续作为调试记录。

## 分段正式训练规则

例如仅为解释参数，若计划 `H=60`，第一段可用 `--epochs 5 --ounet-schedule-epochs 60`，第二段用 `--epochs 10 --ounet-schedule-epochs 60 --resume .../last.pt`。每段完成后可将 `last.pt` 复制为候选权重，再用独立验证集选模；入口没有自动 best 保存功能。

正式轮数、batch 和验证方案需要结合服务器短训情况确定。正式命令不带 `--max-batches`，不使用本次调试 checkpoint，也不覆盖其他四模型目录。

## 本机验证

- Python 3.10 环境、torch `2.5.1+cu121`、ocnn `2.3.3`，使用已有隔离依赖目录，未安装新依赖。
- 新增 5 个测试通过：公式与边界、分段恢复与连续训练逐项一致、配置/状态错误拒绝、调试转正式拒绝、CLI 参数解析和校验。
- 11 个现有回归测试通过：SPU 原 cosine 初始值与共同输入损失、共同训练子集及恢复配置、V3 点数/传输协议。
- 真实 CUDA 短训：本机 RTX 4070 Laptop 8 GB，batch=8、workers=0，2 轮各 3 个 batch，损失有限并正常保存；调度 LR 依次为 `0.0005`、`0.000492493711467861`，第二轮后保存 `0.0004849746889841331`。
- 真实 CUDA 断点恢复通过：从第二轮 checkpoint 接着跑第三轮的 3 个 batch；启动 LR 正确恢复为 `0.0004849746889841331`，第三轮结束后 `next_lr=0.00047744269081074987`。
- 本机调试目录：`outputs/ounet_schedule_check_20261102/real_b8`。这不是服务器 RTX 3080 Ti 的显存测试或正式训练成绩。
