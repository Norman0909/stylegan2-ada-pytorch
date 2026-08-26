# StyleGAN2-ADA 弱类增广实验总结

## 结论一句话

**GAN 生成式增广只对最稀有的类有效(metal bucket),对中等稀有类(cylinder/circle cage)无益甚至微降。**

## 逐类结果(test 集,YOLO yolo26n 逐类对比)

| 类 | 实例数 | ΔmAP50 | ΔmAP50-95 | 判定 |
|---|---|---|---|---|
| metal bucket | 390 | **+3.81%** | **+2.61%** | ✅ 明显提升 |
| cylinder | 526 | -0.45% | +0.25% | ⚠️ 中性 |
| circle cage | 688 | -0.41% | -0.52% | ⚠️ 微降 |
| rov | 799 | — | — | 未做(停) |
| plane | 850 | — | — | 未做(停) |

## 规律

实例数越少,增广收益越大:
- **metal bucket(390)**: 明显提升,mAP50 +3.81%。
- **cylinder(526) / circle cage(688)**: 无益甚至微降。

推断:GAN 生成样本的多样性不足以超过中等稀有类已有的真实样本多样性;只有极度欠采样(390)时,补样本才有效。

## 方法(可复用管线)

1. `extract_crops.py` 提裁剪
2. StyleGAN2-ADA 训 800 kimg
3. `generate.py` 生成
4. `paste_back.py` 按类位置+尺寸贴入(避重叠+尺寸过滤)
5. `build_dataset.py` 组数据集(硬链接)
6. YOLO 逐类对比

## 各实验目录

- `metal_bucket_20260821/`(成功案例)
- `cylinder_20260822/`(中性)
- `circle_cage_20260822/`(微降)

## 建议

- 若目标只是提升 metal bucket,已达成。
- 若需提升 cylinder/circle cage 等中等稀有类,建议改**经典增广**(copy-paste/翻转/颜色扰动),或提高 GAN 样本多样性(更大 kimg、更多真实样本)。
