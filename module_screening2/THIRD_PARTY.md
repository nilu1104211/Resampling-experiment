# 来源与实际适配边界

这些实验用于筛选机制，不是五篇论文的完整复现。所有新增可学习分支随机初始化，只在相同 V3 训练集训练；没有下载外部预训练权重，也没有导入参考仓库的训练数据、CUDA 扩展或运行环境。

本地 ZIP 没有可核验的 Git commit；`source_manifest.json` 记录实际读取文件的 SHA256，不虚构提交号。运行不依赖 `reference2/`，上传本目录即可携带所需实现。

| 来源 | 本次实现 | 与原方法的差异 |
|---|---|---|
| [PUFM](https://github.com/Holmes-Alan/PUFM)；本机 `baseline/PUFM-main` | 原 attention PUFM、原中点候选顺序、5 次 Euler；复制已有 V3 兼容源码 | 仅给速度网络增加默认关闭的 `feature_residual` 接口；通用算子明确使用可重复 portable torch + V3 FPS，所有新对照同样使用该后端。原模型参数严格加载 |
| [Denoise Yourself](https://github.com/jihyeonhur/Denoise-yourself) `util.density`、`DensityData`、`CurvatureData` | 密度分组和法向角变化加权选中点 | 改为固定输出预算；密度中位数分组，低/高组权重比 2:1；输入 XYZ PCA 估计法向，曲率按无向法向夹角；未复制其实例式自监督训练或去噪器 |
| [PointSP](https://github.com/tangsankou/PointSP) `cal_weight`、WRS、FFPS | 输入邻域可靠度，传播到中点两端后做 WRS 或过滤 FPS | 使用原中点池；FFPS 过滤候选的最低 5%，替换 `clock64` 随机起点为 V3 确定性 FPS；WRS 修复 batch=1、合法无放回预算和权重下界。不是原分类网络 |
| PointSP `random_k_neighbors_shape_invariant_perturb` | 邻域位移投影到 PCA 切平面，步长为半位移与局部中位距离的较小值 | 每点 8 个随机邻居生成固定候选池，再选 4N；法向仅来自观测 XYZ；不保留原随机增删点数流程 |
| [DWCNet](https://github.com/keneniwt/DWCNET-Robust-Point-Cloud-Completion-against-Corruptions) `CleanNoisySeparator`、`ImprovedContrastiveLoss` | clean 自注意力/FFN 与 noisy 1/3/5 卷积、正相似度和负 logsumexp | 宽度 64；新增本地观测编码；noisy 卷积沿距离排序的局部邻域运行，避免对任意点顺序卷积；teacher 为对应索引的 clean_input，stop-gradient。只向速度网络最后 128 维特征加残差 |
| [UPP](https://github.com/zhoujiahuan1991/ICCV2025-UPP) Rectify/Completion 机制 | `upp_rectify`：输入位移；`upp_completion`：mask query / cross-attention / coarse+dense anchors | 均为轻量机制适配，未移植 Point-MAE 及官方预训练 Prompt。修正保持全部观测点；补全使用 32 查询和动态 2D grid，预留总初始点数 1/8 给锚点 |
| [SPOFormer](https://github.com/kiddoray/SPOFormer) coarse/keypoint/query 与 projection | `spo_keypoint`：粗候选、独立评分监督、保留锚点；`spo_projection`：邻域投影正则 | 关键点分支为轻量适配；评分增加 BCE，解决仅 argsort 无评分梯度的问题。投影重写为 batched kNN、按 B,N,k,k 取均值，默认 λ=0.1，不能与原 sum reduction 的 λ=1e-5 比较数值 |

UPP、DWCNet、PointSP 本地提供的许可文本保留在 `licenses/`。本地 Denoise Yourself、SPOFormer 和 PUFM 快照未见独立许可文件，保留来源与作者引用；不据此声明这些快照具有某种未给出的许可证。

UPP/SPO 新分支的正向监督仅用 V3 **训练** GT、对应干净观测和区域标签。验证/测试时，网络调用只接收 `[B,N,3]` 观测；GT 与区域标签只传给公共评价器。

依赖和评价部分复用现有 `adaptive_resampling/`、`run_subset_eval.py`，只读调用，未修改其协议。
