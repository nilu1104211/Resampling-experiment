# V3 精确匹配并行加速

新增参数 `--assignment-workers`，将同一个 batch 内独立的 SciPy 精确匹配分给多个 CPU 线程。
默认值是 `1`，保持原有串行行为；使用 `--assignment-workers 4` 开启并行。
没有增加依赖，也没有更换匹配算法、代价矩阵、损失、精度、数据量或训练轮数。
该参数会记录到训练 checkpoint 的 `args` 中，可与原有 V3 权重兼容。

`--workers 4` 控制 DataLoader 的数据读取进程，`--assignment-workers 4` 控制匹配线程，两者作用不同。
batch=4 时先用 4 个匹配线程；batch=1 只有一个匹配任务，增加线程没有帮助。
PUFM++ S1、时间表校准，以及 PUFM 基线/Ours 中的 flow 匹配会使用此实现。
PUFM++ S2、SPU-IMR 和 OUNet 当前的损失不调用这项精确匹配，因此这个参数不会加速它们的训练。

## 服务器测速

先将本地代码提交并推送到 Git。服务器在实验空闲时同步，然后运行与之前相同的 100 个 batch，只增加匹配线程参数并更换输出目录：

```bash
conda activate scdata-v3
cd ~/baseline_compare
git pull
source env_v3.sh

time v3 pufmpp train \
  --stage s1 \
  --device cuda \
  --batch-size 4 \
  --workers 4 \
  --assignment-workers 4 \
  --epochs 1 \
  --max-batches 100 \
  --log-every 20 \
  --output server_runs/speedcheck/pufmpp_b4_w4_match4
```

此前相同 batch/workers 的串行测速耗时为 `1m9.635s`。对比这次的 `real`，才是服务器上整个命令的实际加速幅度。
测速结果仍会标记 `partial_debug_run=true`。正式训练使用原来的完整训练命令，额外添加 `--assignment-workers 4`，并去掉 `--max-batches`。
若要恢复串行匹配，设为 `--assignment-workers 1`。

## 单独比较匹配线程数

```bash
python -B tools/benchmark_v3_assignment.py \
  --device cuda --batch-size 4 --threads 1 2 4 \
  --batches 3 --repeats 3 \
  --output server_runs/speedcheck/assignment_threads.json
```

脚本从真实 V3 训练集固定选择样本，使用 PUFM++ S1 的 midpoint 加噪初始化，比较完全相同的 1024×1024 代价矩阵。
每种线程数都检查匹配索引逐元素相等。只计 CPU 匹配耗时，不计数据读取、初始化、传输、模型计算和保存。

本地测试（RTX 4070 Laptop；32 个逻辑 CPU；PyTorch 2.5.1+cu121；SciPy 1.15.3）结果如下，不能直接当作服务器训练速度：

| 匹配线程数 | 每 batch 匹配时间中位数 | 匹配加速比 |
| --- | ---: | ---: |
| 1 | 422.9 ms | 1.00× |
| 2 | 244.9 ms | 1.73× |
| 4 | 166.0 ms | 2.55× |

并行实现复用了同一个线程池；每个点云仍调用相同的 SciPy 求解器，按原 batch 顺序返回结果。
自动检查覆盖随机点云、重复点/代价相同的情况、不同 batch 大小、CPU/CUDA 对齐结果、损失及 source/target 坐标梯度。

```bash
python -B -m unittest discover -s tests -p 'test_v3_transport.py' -v
python -B compare-change/test_v3_adaptation.py -v
```
