# cylinder 增广结果 (2026-08-22)

## 实验
- StyleGAN2-ADA 800 kimg 生成 1000 张 → 900 张贴入(90% 收率)
- cylinder 526 → 1426 实例
- YOLO yolo26n 逐类对比(test 集)

## 结果

| 指标 | 基线 | 增广 | Δ |
|---|---|---|---|
| cylinder mAP50 | 0.9242 | 0.9196 | -0.0045 |
| cylinder mAP50-95 | 0.4486 | 0.4511 | +0.0025 |
| 平均 mAP50 | 0.9583 | 0.9600 | +0.0017 |
| 平均 mAP50-95 | 0.5361 | 0.5377 | +0.0016 |

## 结论
**中性,无明显提升。** cylinder mAP50-95 仅 +0.25%,mAP50 微降 -0.45%。

可能原因:
1. 526 实例已够,增广边际收益低(对比 metal bucket 390 → 提升明显)。
2. GAN 生成形状多样性不足(竖直 mode collapse)。

## 产物
- 生成: `gen_800k/`(1000 张)
- 贴入: `pasted_full/`(900 张)
- 数据集: `dataset/`(8241 train + 920 val)
- YOLO run: `detect_model/.../runs/detect/sonar_yolo26n_aug_cylinder_20260822/`
