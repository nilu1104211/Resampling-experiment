# 本机实施检查记录

日期：2026-11-05，北京时间。所有运行均为接口排错，无正式训练/模块优劣结论，无服务器连接。

环境：Python 3.13.3、PyTorch 2.7.1+cu118、CUDA 可用。为本机 QA 将 h5py 3.16.0 安装到被忽略的 `tmp/module_screening_deps`；没有修改全局 Python 或服务器 Conda 环境。服务器仍应先运行 README 的 smoke 检查。

| 检查 | 结果与证据 |
|---|---|
| 11 项单元检查 | `tests/test_module_screening.py`：七份物理副本、基线和 reference 哈希不变、关闭特征接口的 backbone AST 等价、共同初始化 SHA、APML 原版/有限梯度、退化协方差、M5 停止保持输入、ATS 非均匀步长/五次调用及区域指标聚合等，全部通过。 |
| 七分支真实 V3 patch | `outputs/module_screening_checks/seven_real_v1/summary.json`：共同正式 PUFM 权重、训练集一个真实 patch、256→1024；M1–M6 有有限非零梯度且完成一次临时 optimizer step；M7 仅推理。M5/M6 参数及 BN 缓冲冻结检查通过。 |
| 七分支整物体尺寸 | `outputs/module_screening_checks/seven_whole_v1/`：相同随机 2048 点、各分支输出 8192 个有限点、恰好五次 flow 调用。输入 SHA 一致，没有使用冻结测试数据或 GT。辅助模块随机初始化；M7 使用合成非均匀网格，仅检查接口。 |
| M1 训练生命周期 | `outputs/module_screening_checks/lifecycle_m1_v2/`：1 epoch、1 batch、batch=1，验证每条件仅 1 条；成功保存 config/manifest、best/last、验证 CSV 和日志。权重带 debug 标记。 |
| M7 校准生命周期 | `outputs/module_screening_checks/ats_lifecycle_v1/schedule.json`：仅一个训练 patch、51 个时间点，完成误差曲线→CDF→六时间点保存。时间表带 debug 标记，绑定基线/数据/源码。 |
| 权重/时间表拦截、读取与续训入口 | `outputs/module_screening_checks/lifecycle_audit.json`：debug 权重和时间表在普通验证中被拒绝，显式 allow-debug 后验证通过；M1 已完成阶段的 resume 入口通过。 |

全量验证、正式 10 轮训练、完整 256 patch ATS 校准和最终冻结 80 物体测试均未执行。没有推送或提交 Git；已有历史文档改动保留。

生命周期排错发现并修复：不同条件的公共评价器会输出不同的区域指标字段，CSV 现按字段并集写出；几何核心指标按七条件等权，区域指标记录实际参与条件数，不补零。
