# 来源、许可证与适配边界

参考代码来自用户放入 `reference/` 的本地快照。`source_manifest.json` 记录每个文件 SHA256；目录没有 `.git` 元数据，来源提交未知。以下链接用于定位项目，本轮没有联网更新这些代码。

| 实验 | 来源与本地入口 | 使用方式 |
|---|---|---|
| 基线 | `baseline/PUFM-main` | 七份源码复制；原网络参数名不变。仅新增可选末层特征残差接口，关闭时原 forward 的 AST 与原文件一致。utils/PointNet2 使用仓库已有 V3 算子兼容层；sklearn 仅在旧便利函数内按需导入。 |
| M1 | [Enhanced_PUFM](https://github.com/Holmes-Alan/Enhanced_PUFM)，`train_pufm_s2.py` | 终点 CD/扰动机制适配，未移植 RIN。参考快照未包含 LICENSE 文件，未推断许可证。 |
| M2 | [APML](https://github.com/apm-loss/apml)，`src/loss/apml_loss.py` | 原文件逐字节复制至 `models/apml_loss.py`，保留其 20 次 Sinkhorn、无梯度匹配与 assignment sum reduction。MIT 原文在 `models/APML_LICENSE`。API 实际参数名是 `min_softmax_value`，并非方案示意的 `p_min`。 |
| M3 | [PU-Gaussian](https://github.com/mvg-inatech/PU-Gaussian)，`model/GaussianDistribution.py`、`GaussianNetwork.py` | 从协方差表示获得启发，重新实现观测 KNN 协方差特征；没有复制高斯生成器或其预训练模型。参考快照附 MIT LICENSE。 |
| M4 | [RepKPU](https://github.com/EasyRy/RepKPU)，`models/decoder/kernel.py:REM` | 提取实际 REM 类到 `repk_rem.py`，保留可变形核、距离影响权重和 RepKPoints 特征融合。不复制 KGM/查询/生成头。MIT 原文在 `models/RepKPU_LICENSE`。 |
| M5 | [ASDN](https://github.com/git-guocc/ASDN)，`models/denoiseCD.py`、`classify.py` | 阅读实际去噪/质量判别流程后自行适配轻量网络；未复制依赖 Lightning/PyTorch3D 的原架构或熵比分类器。训练标签为配对训练干净输入上“提议是否改善”的云级判断，非逐点退化类型。参考快照未包含 LICENSE 文件，未推断许可证。 |
| M6 | [Grad-PU](https://github.com/yunhe20/Grad-PU)，`models/FeatureExtractor.py`、`P2PRegressor.py`、`P2PNet.py` | 三文件分别移植为 `grad_features.py`、`grad_regressor.py`、`grad_p2p.py`，imports 指向本副本算子。Apache 2.0 原文在 `models/GradPU_LICENSE`。 |
| M7 | [Enhanced_PUFM](https://github.com/Holmes-Alan/Enhanced_PUFM)，`train_pufm_s3.py:build_entropic_schedule` | 训练集误差梯形积分→CDF→反演时间表。增加密度下限和单调校验；使用原 PUFM 的单次速度评估，不使用 RIN 自条件或 manifold 后处理。 |

M4 额外适配：保留参考 `k_015_center.ply` 核点；轻量读取器避免生成/写入 reference，并使用种子控制旋转和微扰；对参考 REM 的平方根加 `1e-12` 下限，避免中心重合点反向梯度出现 NaN。半径、邻域数仍按参考配置，新增投影将核点特征均值映射回 PUFM 128 维。训练仍仅优化原 FM 目标，REM 自带 fit/rep loss 未并入目标。

M6 额外适配：最后的距离 ReLU 改成 `Softplus(beta=20)`，避免随机初始化时整个距离头无梯度；训练对象是 PUFM 输出附近的 jitter query，GT 只生成训练距离标签。精修在完整输出上进行，不使用原 Grad-PU 的切块重采样流程；坐标更新按局部 spacing 截断并保持点数。以上均是本实验适配，不代表官方原样配置。

七项互不叠加。M1–M4 的额外训练、M5/M6 的额外参数与耗时、M7 的校准样本和时间表均需分别报告。
