# circle cage 增广结果 (2026-08-25)

## 实验
- StyleGAN2-ADA 800 kimg 生成 1000 张 → 686 张贴入(68.6% 收率)
- circle cage 688 → 1374 实例
- YOLO yolo26n 逐类对比(test 集)

## 结果

| 指标 | 基线 | 增广 | Δ |
|---|---|---|---|
| circle cage mAP50 | 0.9146 | 0.9105 | -0.0041 |
| circle cage mAP50-95 | 0.5106 | 0.5055 | -0.0052 |
| 平均 mAP50 | 0.9583 | 0.9564 | -0.0019 |
| 平均 mAP50-95 | 0.5361 | 0.5292 | -0.0069 |

## 结论
**轻微下降,无益。** circle cage 增广未带来提升,反而微降。

累计规律(metal bucket/cylinder/circle cage):实例越少越受益,metal bucket(390)明显提升,cylinder(526)/circle cage(688)无益甚至微降。

## 产物
- 生成: `gen_800k/`(1000 张)
- 贴入: `pasted_full/`(686 张)
- 数据集: `dataset/`(8027 train + 920 val)
- YOLO run: `detect_model/.../runs/detect/sonar_yolo26n_aug_circle_cage_20260822/`
