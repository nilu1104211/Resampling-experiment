# 第二批独立 PUFM 模块筛选：15 轮

2026-11-06。十个候选机制，加一份原 PUFM 控制组，共 **11 份物理独立源码**。每份有独立模型文件、`screening.py` 和配置。进程之间不共享 `models` 包；没有链接，也不从其他实验副本或 `reference2/` 导入网络。

本目录只存源码、配置、检查和说明；没有数据、权重、日志、notebook、旧 CUDA 扩展、图片或参考仓库整份拷贝。原始 `baseline/`、上一轮 `module_screening/`、`reference2/` 和公共 V3 协议不修改。

## 实际实验与控制组

| 独立副本 | 默认候选 | 额外变体 / 说明 |
|---|---|---|
| `control/PUFM-main` | B0 原始冻结 PUFM | B-extra：相同 15 轮原 FM 续训 |
| `pointsp_wrs/PUFM-main` | 可靠度加权中点候选抽样 | 相同候选池的 uniform random 对照 |
| `pointsp_ffps/PUFM-main` | 过滤最低 5% 可靠度候选，再 FPS | 使用确定性起点，不使用 clock64 |
| `dy_density/PUFM-main` | 低密度分组加权 | 高密度分组加权单独运行，均为 2:1 权重 |
| `dy_curvature/PUFM-main` | 邻域无向法向夹角加权 | 法向仅由输入 XYZ 的 PCA 估计 |
| `pointsp_lgp/PUFM-main` | 邻域切平面传播初始化 | 固定生成候选与选取数量 |
| `upp_rectify/PUFM-main` | 轻量输入位移适配 | 零位移初始化；训练干净对应点监督；不删除输入点 |
| `dwc_nmm/PUFM-main` | clean/noisy 分离 + 对比监督 | 无对比损失消融单独训练 |
| `upp_completion/PUFM-main` | mask query 辅助补全锚点 | 轻量机制适配，未加载 Point-MAE |
| `spo_keypoint/PUFM-main` | 粗关键点、监督评分和锚点 | 显式评分 BCE，保证评分网络有梯度 |
| `spo_projection/PUFM-main` | 单步终点估计上的投影正则 | FM + λ·projection；没有额外终点 CD |

UPP 和 SPO 的新分支是为 PUFM 编写的轻量机制适配，**不能称为原论文完整模块复现**。准确来源、简化、数值处理与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)，本地来源指纹见 `source_manifest.json`。

默认一键队列共 **23 个阶段**：

1. 一组真实 patch、重复前向、加载复现、禁用模块一致性与整物体形状检查。
2. B0 完整验证，加 7 项免训练采样验证，共 8 项验证。
3. 14 项各 15 轮训练：B-extra、uniform、WRS、FFPS、密度低/高、曲率、LGP、Rectify、NMM、NMM 无对比、UPP 补全、SPO 关键点、SPO 投影。

十个机制之外增加的训练用于检验抽样权重、密度方向和对比监督。每阶段完成并核验后才启动下一项，失败立即停止。**本队列只筛选验证集，不自动运行冻结整物体测试。**

## 已固定的公平性条件

- 原始 attention PUFM；所有实验从同一个 `pufm_baseline_b8_w0_a8_fused/epoch_0030.pt` 开始。SHA256：`c602029630079252fcd249078877ebaae1be083c99e829a82c8e15aafd01887e`。这是已有训练历史的基线，再续训 15 轮。
- 训练：V3 mixed、同样的 797 对象 × 每对象 32 patch、同样的在线退化与旋转。训练选择指纹自动记入 manifest。
- 训练输入/GT：**256 / 1024**；验证同样 256→1024；显式冻结整物体测试保持 **2048→8192**。
- 每项训练 **15 轮、batch=8、Adam、lr=1e-5、seed=21**，workers=0、assignment_workers=8。正常完整训练每轮 3188 step，共 **47820 step、382560 patch**。新增分支与 PUFM 联合训练，没有隐含预训练或额外 warmup。
- FM 目标、匹配算法、时间分布、训练初始化扰动 σ=0.01 相同。辅助监督及系数在各自配置中预先固定，不用测试结果调整。
- 推理统一 **5 次 Euler 速度评估**，不增加输出点、不增加观测、不传 GT、真值法向、干净参考或退化标签。
- 验证：100 个 PU1K 对象 × 4 patch × 7 条件 = 2800 条；先物体内 patch 平均，再物体平均，七条件等权。第 **5/10/15** 轮全验证，以相同 CD-L1 最小规则选 best；同时保留全部条件、HD、F-score 和区域指标。
- 无学习参数的采样先用 B0 免训练验证，然后各自续训 15 轮与 B-extra 比较。uniform 与 WRS 的差别只在权重。所有基线都在本轮同环境重新运行。
- 普通候选池保留原 PUFM 的 **8N 中点候选 → 4N 选点**；不偷偷强制保留所有原始点。补全分支显式保留 4N 总预算中的 1/8 锚点，并记录保留数；不在混合后又被 FPS 丢弃。
- 统一 portable torch 邻域算子和 V3 确定性 FPS；CUDA 运行默认要求 NVRTC FPS 成功，不静默换后端。关闭 TF32，启用确定性算法，记录 Torch/CUDA/GPU/后端。不会安装参考仓库旧扩展或修改 Conda 环境。
- 采样使用独立、输入绑定的 CPU generator；不改变共用 FM 噪声/时间 RNG。验证只在独立 Torch RNG scope 中运行。
- 启动核对权重、metadata、实际 100 个验证 H5 的哈希；每次验证保存输入/预测 SHA256，汇总时拒绝输入、环境或协议不一致的比较。

这些措施保证本轮比较口径可核查；**不同 GPU/Torch 之间逐位一致仍需实际验证**，不能据此宣称上一轮跨环境复算问题已解决。15 轮属于单 seed 筛选，负结果不能直接否定原论文方法。

## 服务器一键启动

先把本目录推送到服务器，再在已有环境运行：

```bash
conda activate scdata-v3
cd /home/zk/baseline_compare
git pull --ff-only origin main
bash module_screening2/start_15epochs.sh
```

脚本后台运行，打印 PID、日志和结果位置。默认结果目录：

```text
/home/zk/baseline_compare/server_runs/patch32/module_screening2/sequential15_v1/
  queue.json
  checks/
  configs/
  logs/
  runs/<阶段名>/
  comparison.csv
  comparison.json
```

默认数据、权重位置沿用服务器既有目录。如路径不同可设置 `V3_DATA_ROOT`、`PUFM_BASELINE_CHECKPOINT`、`MODULE_SCREENING2_OUTPUT`。正式队列检查输出文件系统至少有 12 GiB 可用空间；不会自动清理任何旧文件。

```bash
tail -f server_runs/patch32/module_screening2/sequential15_v1.console.log
```

中断后，确认原队列及其训练子进程已退出，使用原选项恢复：

```bash
bash module_screening2/start_15epochs.sh --resume
```

已完成任务核验后跳过；训练从自己的 `last.pt` 恢复；权重、代码、数据或配置改变会拒绝续跑。同一输出有进程锁，禁止并发覆盖。训练在第 0 轮先保存可恢复状态；若在该状态写入前启动失败且没有 `last.pt`，保留失败目录并选新输出。不自动删除失败证据。

## 查看计划与单独运行

```bash
# 不创建文件、不读取权重和数据。
python -B module_screening2/train_all.py --dry-run

# 只训练 WRS 时，自动加入 B0、15 轮 B-extra 和 uniform random 对照。
python -u -B module_screening2/train_all.py --tasks pointsp_wrs --fused-optimizer \
  --output server_runs/patch32/module_screening2/wrs15_v1

# 独立真实数据接口检查；不是效果成绩。
python -B module_screening2/check.py --real-patch --whole-shape --device cuda \
  --output outputs/module_screening2_checks/server_v1

python -B -m unittest discover -s module_screening2/tests -p 'test_*.py' -v
```

`--limit-batches` / `--limit-samples` 只用于 debug，配置、权重、汇总会标记 partial，不能冒充完整实验。`--skip-checks` 仅供已在同环境单独通过检查后使用。

每项训练保存 `best.pt`、`last.pt`、`best.json`、`manifest.json`、`train.jsonl` 和第 5/10/15 轮逐样本/逐物体指标；不保存全部 epoch 权重。学习模块在 `diagnostics.json` 记录位移、特征残差或锚点数。推理时延包含模块与初始化，排除首次 warmup、数据读取和指标计算。

## Git 只上传本次新目录

当前检查分支为 `main`，远程为 `origin`。在本机 PowerShell：

```powershell
cd D:\dkwork\baseline_compare
git add -- module_screening2
git diff --cached --stat -- module_screening2
git commit --only -m "Add independent 15-epoch PUFM module screening" -- module_screening2
git push origin main
```

`commit --only` 限定提交本次目录，避免夹带此前尚未提交的改动。不使用 `git add .`；无需上传 `reference2/`。数据、权重、结果、缓存和日志均被忽略，本目录也拒绝作为训练结果输出位置。
