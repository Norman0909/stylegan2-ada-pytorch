# Phase 0：原版 vs 新版直接对比

日期：2026-09-04
状态：已完成，暂不进入新的 GAN 长训

## 1. 实验范围

本阶段对比：

- 原版：`experiments/cylinder_20260822/gen_800k` 中的 1,000 张 `seed0000`–`seed0999`。
- 新版：UATD cylinder 的 target-context、128×128、单通道、batch 32、ADA 配置，累计有效约 800 kimg 的最终 checkpoint。
- 新版 checkpoint：`experiments/uatd_cylinder_context_20260902_128x1_b32_ada_bgc_k800/gan_runs/00002-uatd_cylinder_context_128-auto1-kimg200-batch32-ada-bgc-resumecustom/network-snapshot-000200.pkl`。
- 两边均使用固定 seeds 0–999、`trunc=1`、`noise-mode=const`，并统一在 UATD processed test split 上验证 YOLO。

这不是严格意义上的单变量消融：原版使用 bbox padding 1.6、黑色方形补边和 RGB；新版使用 `context_scale=5.0`、edge-replicate 补边和单通道 L。因此下面结论是“端到端旧流程 vs 新流程”，不能把全部差异归因于 context scale。

## 2. GAN 生成结果

| 指标 | 原版 800k | 新版有效 800k | 观察 |
|---|---:|---:|---|
| 有效图像数 | 1000 | 1000 | 满足对比条件 |
| 像素 mean / std | 2.671 / 6.563 | 3.183 / 8.158 | 新版更亮、动态范围更大 |
| p90 / p99 | 7 / 34 | 6 / 44 | 新版高亮尾部更重 |
| 像素 ≤1 占比 | 68.91% | 64.91% | 新版背景稀疏度下降 |
| 阈值 >8 前景占比（均值/中位数） | 8.15% / 6.98% | 7.36% / 1.89% | 新版个体差异更大 |
| 空前景图占比 | 0% | 0.1% | 新版有 1 张空前景图 |
| 阈值前景 bbox 面积（中位数） | 24.13% | 95.36% | 新版阈值掩膜常扩散到整幅 patch |
| 随机 pair RMS（均值） | 5.759 | 6.825 | 新版全局变化更大 |
| 最近邻 RMS（均值） | 0.834 | 0.625 | 新版存在更相似的样本簇 |

从固定种子 contact sheet 看，新版会生成较多中心稀疏亮斑和背景纹理；目标及声影的完整几何形态没有稳定地呈现。全局统计显示新版并非简单复制旧图，但不能据此判定其目标结构更真实。

## 3. 源 patch 分布

新 manifest 共 526 张 patch，目标长边固定约为 patch 的 20%，目标框面积均值约 2.52%（p10–p90 为 1.21%–3.59%），当前没有显式的声影 mask 标签。

| 指标 | 原版 crop | 新版 context patch |
|---|---:|---:|
| patch mean / std | 3.159 / 7.545 | 3.261 / 8.076 |
| 阈值 >8 前景占比中位数 | 8.50% | 2.70% |
| 空前景占比 | 0% | 0.19% |
| 随机 pair RMS（均值） | 6.434 | 7.061 |

新版 patch 本身已经具有更大的局部强度变化和更小的典型阈值前景；这与生成结果中“少量目标亮斑 + 大面积上下文”的现象是一致的。

## 4. 贴回流程检查

使用相同的 `aug/paste_back.py` 设置和随机种子后：

| 指标 | 原版生成图 | 新版生成图 |
|---|---:|---:|
| 接受 | 900 | 723 |
| 跳过 | 100 | 277 |
| 贴回后 bbox 宽中位数 | 41 px | 79 px |
| 贴回后 bbox 高中位数 | 38 px | 69 px |
| patch 亮度均值 | 9.1 | 6.6 |

新版的阈值 `gray > 8` 把上下文纹理也当作前景，导致 bbox 过大、亮度偏低、可接受样本减少。这是当前最明确的工程问题。继续增加 GAN kimg 不能直接修复这个解析/贴回错配。

## 5. 下游 YOLO 结果

三组权重使用同一 UATD processed test split、`imgsz=640`、`batch=16` 验证：

| 模型 | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| 原始 baseline | 0.95831 | 0.53607 | 0.95825 | 0.94654 |
| 原版增强 800k | 0.96004 | 0.53771 | 0.95682 | 0.94272 |
| 新版增强有效 800k | 0.96184 | 0.52821 | 0.96078 | 0.94885 |

相对原版增强：

- mAP50：`+0.00180`，略有提升；
- mAP50-95：`-0.00949`，明显下降；
- cylinder 类 mAP50：`0.91963 → 0.91569`；
- cylinder 类 mAP50-95：`0.45110 → 0.44501`。

因此当前不能把新版判定为优于原版。mAP50 的小幅提升不足以抵消定位质量（mAP50-95）和 cylinder 类指标的下降。

## 6. 运行备注

- 新版 GAN 日志已正常结束；最终单次 resume 日志显示 `kimg 200.0`，有效累计预算为原始 400 + 两次各 200。
- 当前 128×128 单通道训练峰值显存约 6.22 GB，没有 OOM 证据，因此暂不启动 256×256。
- 训练期间环境中的 Polars 因 `unknown feature flag: 'sse3'` 导致 YOLO 的结果绘图阶段报错；训练、`best.pt` 保存和统一 test 验证均已完成，故不影响本次数值结果。结果文件仍保存在 `downstream_yolo/yolo_runs/new_augmented_effective800_retry/`。
- 由于训练配置 `metrics=[]`，本阶段不宣称 FID 或可比 FID。

## 7. 下一步建议

下一步应先做“小范围贴回适配实验”，保持 GAN checkpoint、128×128 单通道和 YOLO 设置不变：

1. 给新生成图增加独立的目标/声影 mask 解析或显式 alpha/metadata；不要继续沿用只按 `gray > 8` 的旧前景提取器作为唯一依据。
2. 用固定 1,000 seeds 重新检查接受率、bbox 分布、亮度和目标尺寸；先让贴回统计接近真实 patch/原版增强的范围。
3. 适配通过后，只重跑一次下游 YOLO，确认 mAP50-95 是否恢复；若仍低于原版，再淘汰当前 `context_scale=5.0` 表示。
4. 若适配后有效，再进行计划中的 `context_scale=3.0`、灰度 128、batch 32、约 300 kimg screen。256 分辨率、真三通道和网络结构改动继续后置。

当前建议：**不补训、不启动 256、不先改网络；先修复新版生成 patch 的 mask/贴回适配。**

## 8. 复核入口

- 对比清单：[comparison_manifest.json](comparison_manifest.json)
- GAN/patch 统计：[diagnostics/image_stats.json](diagnostics/image_stats.json)
- 掩膜质量：[diagnostics/mask_quality.json](diagnostics/mask_quality.json)
- 下游验证：[downstream_yolo/validation_results.json](downstream_yolo/validation_results.json)
- 可视化 contact sheet：[diagnostics/contact_sheets/old_top_new_bottom_seed0000-0063.png](diagnostics/contact_sheets/old_top_new_bottom_seed0000-0063.png)
