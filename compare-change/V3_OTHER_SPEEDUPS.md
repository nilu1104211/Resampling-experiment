# V3 后续加速与 fused 优化器

当前训练已按每对象 32 个 patch 配置，详见 [32-patch 训练配置](V3_TRAINING_32.md)。下面的优化器历史测速仅用于说明参数更新开销。

## 已加入的第二项：fused 优化器

在训练命令中添加 `--fused-optimizer`，启用 PyTorch 自带的 CUDA fused AdamW/Adam。
对比网络保留 AdamW，PUFM 基线/Ours 保留 Adam，学习率、权重衰减和数据均保持原设置。
默认关闭，不需要安装新依赖。开关会记录到 checkpoint 的 `args` 中。

这次优化与精确匹配并行不同：融合浮点运算可能有细小舍入差异，不保证训练轨迹逐位相等。
CPU/CUDA 匹配结果不受此开关影响。正式比较时记录并统一各方法的优化器设置，并检查验证指标。
实现会在加载旧 checkpoint 后重新应用 fused 标志，迁移旧 CPU step 计数器，保留已有动量和更新步数。
未传开关时，恢复训练保留 checkpoint 中已有的优化器实现设置。

本地测试使用 PUFM++ 实际的参数形状和固定合成梯度，预热后计时：默认 AdamW 更新中位数 **49.6 ms**，fused **21.8 ms**。
该结果只覆盖参数更新，不包含 forward、backward、匹配、读取和保存，不能当作整轮训练加速比。
RTX 4070 Laptop / PyTorch 2.5.1+cu121 下，Adam/AdamW 多步更新与普通实现通过数值容差检查，旧 checkpoint 转 fused 恢复检查通过。
另外完成 batch=4 的 PUFM++ S1 连续 6 步前向、反向和更新，损失均为有限值；这些检查不代替完整训练的验证指标。

将本地更新提交推送后，在服务器空闲时运行：

```bash
conda activate scdata-v3
cd ~/baseline_compare
git pull
source env_v3.sh

time v3 pufmpp train \
  --stage s1 --device cuda \
  --batch-size 4 --workers 4 \
  --assignment-workers 4 \
  --fused-optimizer \
  --epochs 1 --max-batches 100 --log-every 20 \
  --output server_runs/speedcheck/pufmpp_b4_w4_match4_fused
```

与只加 `--assignment-workers 4` 的测速比较，才能知道优化器额外节省多少时间。
原始串行匹配和默认优化器的同配置结果为 `1m9.635s`。这条命令仍是测速，正式训练去掉 `--max-batches`。

## 尚未启用的其他位置

| 位置 | 当前证据与可尝试方法 | 需要核对的内容 |
| --- | --- | --- |
| FPS 最远点采样 | 现有实现每次最多循环 1024 次；服务器 20-batch profile 中累计约 1.855/17.113 秒。可接入 CUDA 采样算子。 | 保留原采样起点、等距时的索引规则和重复点处理；核对 forward/梯度及服务器编译环境。当前没有替换。 |
| PUFM++ attention | RIN 中手写了 QK、softmax、AV，可比较 PyTorch SDPA。 | 比较实际网络速度、显存、输出和梯度；融合后可能有舍入差异。当前没有切换。 |
| GPU 混合精度 | 可尝试 BF16/FP16，降低部分网络计算和激活开销。 | 保留匹配/几何损失所需精度；检查 NaN、梯度和验证指标。当前仍为原 FP32 流程。 |
| 数据读取 | 已有 4 个 worker 和 persistent workers；可对比 2/4/8 个 worker、pinned memory 和异步传输。 | 看加速后的 DataLoader 等待是否仍明显；CPU 还要同时运行匹配线程，不能只增加进程数。当前没有改动。 |
| batch 大小 | 已测 batch=1~4，batch=4 样本吞吐最高。 | 只有显存有余量时才继续测 6/8；同时记录学习率、每轮更新步数和验证指标。 |
| 每对象 patch 数 64→32 | 已统一调整为 797×32=25,504 条/epoch；固定 batch 下每轮步数减半。 | 当前所有训练入口默认 32，验证/测试不变；需记录预算并检查验证指标。 |

profile 数字包含启动等开销，且 CUDA 异步运行会影响 CPU 归因；不能把各项收益直接相加。
建议先比较“原始 / 并行匹配 / 并行匹配+fused”三个 100-batch 测速，再对加速后的版本重新 profile。

官方实现说明：[PyTorch 2.5.1 AdamW](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py)、[PyTorch 2.5.1 functional/SDPA](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/functional.py)。

```bash
python -B -m unittest discover -s tests -p 'test_v3_optimizer.py' -v
```
