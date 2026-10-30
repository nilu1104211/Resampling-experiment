# V3 点云重采样对比实验

本仓库包含 PUFM 基线、Ours、PUFM++、SPU-IMR 和 OUNet 的 V3 实验入口。

- 完整运行步骤：[服务器实验运行说明](服务器实验运行说明.md)
- 训练预算和选模：[训练安排_V3_建议版](训练安排_V3_建议版.md)
- 数据协议：[数据使用说明_V3](数据使用说明_V3.md)

## 服务器开始运行

进入服务器上的仓库目录。下面按当前目录 `~/baseline_compare` 举例；若克隆成其他名称，进入实际克隆目录即可。

```bash
cd ~/baseline_compare
git pull

# 第一次安装时执行；已安装则直接激活。
conda create -n scdata-v3 python=3.10 -y
conda activate scdata-v3
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements_v3.txt

# 每个新终端重新激活环境、加载本文件。
source env_v3.sh
export CUDA_VISIBLE_DEVICES=0
```

PyTorch wheel 采用项目已验证的版本，服务器 NVIDIA 驱动需支持对应 CUDA。仓库的 `requirements_v3.txt` 不包括 PyTorch / torchvision。

## 数据单独传输

`adaptive_resampling_data_v3.tar` 单独上传，不通过 Git 提交。先查看包内顶层，再解压，使最终存在：

```text
baseline_compare/
├── adaptive_resampling/                         # 数据读取代码，随 Git 同步
├── adaptive_resampling_data_v3/                 # 解压的数据，不进 Git
│   ├── metadata/base_manifest.json
│   ├── train_bases/
│   └── processed/
├── compare-change/                             # 三个对比方法
├── our/PUFM-main/                              # 共同 PUFM 基线与 Ours
├── tools/evaluate_v3_predictions.py
├── examples/use_v3_data.py
├── env_v3.sh
└── requirements_v3.txt
```

```bash
tar -tf adaptive_resampling_data_v3.tar | head -20
# 若顶层就是 adaptive_resampling_data_v3/，直接解压到仓库根目录。
tar -xf adaptive_resampling_data_v3.tar

# 数据也可以放到其他盘；默认是仓库根目录下的同名文件夹。
# export V3_DATA_ROOT=/data/adaptive_resampling_data_v3

python -B examples/use_v3_data.py --numpy-only
python -B examples/use_v3_data.py --workers 2 --batch-size 2
```

上述检查、常规训练、预测和 CD/HD 评价不需要原始网格。`v3 pufmpp check-data` 还会检查官方 XYZ，只有补齐下面的原始文件后才运行它。

## 网络入口

当前所有训练入口默认每个对象使用 **32 个 patch**，共 **25,504 条/epoch**；训练对象仍为 797 个。已有数据包可以继续使用；`--batch-size` 保留各命令设置。说明与检查命令见 [32-patch 训练配置](compare-change/V3_TRAINING_32.md)。

SPU-IMR 主对比使用 `--spu-profile shared-v3`（默认），读取与其他方法完全相同的 256 点训练观测与 1024 点 GT，保留其输入内部的自监督目标。局部输出为 1024 点，整物体为 2048→8192。之前的 native-1024 训练入口已禁用；60 轮训练命令及加速说明见 [最新更新说明](docs/测试加速与SPU适配更新_2026-10-30.md)。

```bash
v3 pufmpp --help
v3 spu-imr --help
v3 ounet --help
ours --help
```

`v3` 和 `ours` 是 `source env_v3.sh` 定义的 Bash 函数。三个对比方法分别调用 `compare-change` 下的 `v3.py`，PUFM 基线与 Ours 调用 `our/PUFM-main/adaptive_v3.py`。完整训练、候选保存、PUFM++ 时间表、预测和评价命令见运行说明。

PUFM++ S1、PUFM 基线与 Ours 的精确匹配可加 `--assignment-workers 4` 并行处理。默认仍为串行；测速命令和适用范围见 [V3 匹配加速说明](compare-change/V3_MATCHING_SPEED.md)。

训练还可加 `--fused-optimizer` 使用 CUDA 融合优化器。它默认关闭，可能有细小浮点舍入差异；测速命令与其他候选优化见 [V3 后续加速说明](compare-change/V3_OTHER_SPEEDUPS.md)。

## 官方 XYZ 与 P2F 网格

这些原始文件也单独上传，保留根目录以下的层级：

```text
data/
├── PU1K/PU1K/test/input_2048/     # 官方 XYZ：input_2048/ 与 gt_8192/
├── PU1K_raw_meshes/              # PU1K 网格，保留原 train/test 层级
└── ModelNet40/ModelNet40/         # ModelNet40 网格，保留原类别/test 层级
```

```bash
export V3_RAW_ROOT="$V3_CODE_ROOT/data"
# 若原始数据在其他盘，改为实际原始数据根目录。
v3 pufmpp check-data
```

读取器会把记录的旧 Windows 原始路径映射到 `V3_RAW_ROOT`，不重写数据。缺少原始网格时，公共评价命令先不加 `--p2f`。

## 本地提交与服务器同步

本地查看并提交这次整理的代码后推送；服务器再 `git pull`。数据包、数据目录、新权重、运行输出和 Python 缓存已有 `.gitignore` 规则。历史提交中已跟踪的文件不会因新忽略规则而自动移除。

无需上传 Windows Conda 环境或 `network_adaptation/deps`。服务器依赖在自己的 Conda 环境中安装。
