# 段错误排查与保留权重续训

## 当前证据

日志在 `epoch: 4, step: 1100` 后出现“段错误（核心已转储）”，随后才有 `resource_sharer` 的连接重置。
epoch 从 0 开始，因此已完成 4 轮，第 5 轮中途退出。应先检查 `last.pt` 的 `epoch`；若为 3，前 4 轮已保存，可从第 5 轮重新开始。
当前代码只在一轮结束时保存，没有保存第 5 轮中途的 1100 步。

段错误是底层本地代码崩溃的信号；后面的多进程连接重置可能是进程退出后的连带异常。
现有日志不足以确定是数据加载、CPU 匹配、优化器、CUDA 库还是其他底层模块。此前 100-batch 测速不能代替长训稳定性验证。

## 新增的恢复选项

新增 `--no-fused-optimizer`，可在恢复旧 fused checkpoint 时明确关闭 fused 更新。
单纯省略 `--fused-optimizer` 不会覆盖 checkpoint 恢复的设置。新开关保留模型权重、动量和 step，只调整优化器执行实现及 step 计数器所在设备。
它与 `--fused-optimizer` 互斥。三种对比网络、PUFM 基线和 Ours 的训练/收益训练入口共用该恢复逻辑。

本地测试覆盖 Adam/AdamW 的 fused→普通实现恢复：动量和计数器保持原值，随后更新通过数值容差检查。
这只验证恢复选项正确，不代表已定位或修复服务器的段错误。

## 先确认已保存的轮次

本地推送本次更新后，在服务器执行：

```bash
conda activate scdata-v3
cd ~/baseline_compare
git pull origin main
source env_v3.sh

python -B -c 'import torch; c=torch.load("server_runs/patch32/pufmpp_s1_b8_a8_fused/last.pt", map_location="cpu", weights_only=False); print("已保存完整轮数:", c["epoch"]+1); print(c.get("result")); print(c.get("training_profile"))'
df -h . /dev/shm
```

若权重读取报错，保留原文件并先检查具体错误；不要直接改成从随机初始化训练。
如需原始内核线索，可尝试 `journalctl -k -n 80 --no-pager`；部分服务器需要管理员权限才能读取相关日志。

## 保守配置恢复第 5 轮并记录崩溃栈

确认已保存 4 轮后，用独立输出目录先完成第 5 轮，保留原来的 `last.pt`：

```bash
export PYTHONFAULTHANDLER=1
set -o pipefail
mkdir -p server_runs/patch32/pufmpp_s1_recovery

{ time v3 pufmpp train \
  --stage s1 --device cuda \
  --train-patches-per-object 32 \
  --batch-size 8 --workers 0 \
  --assignment-workers 1 --no-fused-optimizer \
  --resume server_runs/patch32/pufmpp_s1_b8_a8_fused/last.pt \
  --epochs 5 --log-every 100 \
  --output server_runs/patch32/pufmpp_s1_recovery; } \
  2>&1 | tee -a server_runs/patch32/pufmpp_s1_recovery/recovery.log
```

这里的 `--epochs 5` 表示总计完成 5 轮；如果 checkpoint 已完成 4 轮，只会再跑 1 轮。
`workers=0` 取消 DataLoader 子进程，`assignment-workers=1` 使用串行匹配，并显式关闭 fused，降低排查时的并发复杂度。
batch、训练对象、32-patch 子集、学习率和模型结构保持原设置。这个配置可能更慢，目的是排查和恢复，不是新的最快配置。
启动输出的 `runtime` 应包含 `start_epoch_zero_based: 4`、`loader_workers: 0`、`assignment_workers: 1`、`fused_optimizer: false`。

若再次段错误，`PYTHONFAULTHANDLER=1` 会尝试输出故障发生时的 Python 调用栈，日志会写入 `recovery.log`，便于区别主进程和数据子进程崩溃。
见 [Python faulthandler 官方说明](https://docs.python.org/3.10/library/faulthandler.html)。

第 5 轮完成后，继续训练时改为 `--resume server_runs/patch32/pufmpp_s1_recovery/last.pt --epochs 60`，保留同一恢复目录与配置。
若要逐项恢复并行加速，应先比较完整日志和稳定性，再一次只恢复一个选项，避免立即重复同一组高并发设置。
