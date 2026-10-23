# 当前训练预算：每对象 32 个 patch

按当前实验设置，所有 V3 对比网络、PUFM 基线和 Ours 默认从每个训练对象已有的 64 个基础 patch 中，选择固定的 32 个。
这个数是每个训练对象的 patch 数，与每步输入几个样本的 `--batch-size` 分开设置。

| 项目 | 原配置 | 当前配置 |
| --- | ---: | ---: |
| 训练对象 | 797 | 797 |
| 每对象 patch 数 | 64 | 32 |
| 每 epoch 训练样本数 | 51,008 | 25,504 |
| batch=2 更新步数 | 25,504 | 12,752 |
| batch=4 更新步数 | 12,752 | 6,376 |
| batch=8 更新步数 | 6,376 | 3,188 |

相同 batch 和轮数下，训练步数减半；完整耗时仍包括加载、保存、验证等开销。
batch size、学习率、网络点数和训练轮数保留原设置。这个子集是当前完整训练预算，不会单独触发 `partial_debug_run`。

## 同步后直接使用

本地将新增和修改文件一并提交推送，服务器在训练空闲时更新代码：

```bash
conda activate scdata-v3
cd ~/baseline_compare
git pull
source env_v3.sh

python -B -c 'from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset; d=AdaptiveResamplingDataset(); print(d.training_profile); print("samples:", len(d), "steps at batch=4:", (len(d)+3)//4); d.close()'
```

应显示 `patches_per_object: 32`、`objects: 797`、`samples: 25504`，batch=4 时为 `6376` 步。
`v3 pufmpp`、`v3 spu-imr`、`v3 ounet` 和 `ours` 均默认使用同一选择规则。命令中也可以显式写 `--train-patches-per-object 32`。

例如，用独立目录重新测速：

```bash
time v3 pufmpp train \
  --stage s1 --device cuda \
  --train-patches-per-object 32 \
  --batch-size 4 --workers 4 \
  --assignment-workers 4 --fused-optimizer \
  --epochs 1 --max-batches 100 --log-every 20 \
  --output server_runs/patch32/speedcheck/pufmpp_b4
```

这条仍是 100-batch 测速。正式训练使用新的 `server_runs/patch32/...` 目录，去掉 `--max-batches`，轮数使用既定实验计划。
固定 100 个 batch 的工作量并没有因为每对象 patch 减半而减半；减少的是完整一轮的步数。

## 数据与复现

- 原数据包继续保留每对象 64 行；读取时选择 32 行，不必重新生成、压缩或上传数据。
- 不修改 `generation_config.json` 和 `base_manifest.json`。它们继续描述磁盘上已有的 64-patch 基库。
- 选择由对象 ID 和固定 V3 种子决定，与网络、训练 seed、epoch 和 worker 数无关；每个对象保留全部 32 行，不会变成只训练一半对象。
- 保留原始全量索引供在线退化使用，避免子集重编号改变同一 patch 的退化。patch ID、坐标和监督标签保持对应。
- 验证集、测试集和公开官方 XYZ 的读取范围保持原样。
- checkpoint、教师缓存和 PUFM++ 时间表记录训练选择的数量与 SHA256。所有方法使用同一 `training_profile`。
- `--train-patches-per-object 0` 可读取全部已有 patch，用于重现旧全量配置。

## 旧权重和教师缓存

从 64-patch 配置切换到 32-patch 配置时，使用新的输出目录开始实验；旧 `--resume` 会拒绝直接混用预算。
若明确要继承旧权重，使用 `--checkpoint` 开始一个新阶段，并记录它来自原全量训练；公平从头比较应让所有方法使用相同的 32-patch 预算。
Ours 的教师缓存也记录同一子集；64-patch 配置产生的旧缓存需在新配置下重建。

已检查：797 个对象各 32 个 patch、两类入口子集指纹相同、抽样原始/子集在线样本及监督逐元素一致、workers=0/2 persistent workers 跨 epoch 一致、验证/测试不变、checkpoint 保存和恢复检查。
