# 实验: metal_bucket 数据增广 (2026-08-21)

用 StyleGAN2-ADA 对 UATD 最稀有类 metal bucket(390 实例)做生成式数据增广。

## 目录结构

```
metal_bucket_20260821/
  crops/            # 390 张真实金属桶裁剪(128×128 RGB)
  gan_dataset.zip   # dataset_tool.py 打包的训练数据
  gen_100k/         # 100 kimg 生成(质量差,仅作中间对照)
  gen_800k/         # 800 kimg 生成(最终,1000 张)
  pasted_full/      # 697 张贴入样本(images + labels + manifest.json)
  dataset/          # 增广数据集: 原 train(7341) + 贴入(697) = 8038; val 920
    data_aug.yaml   #   path/train/val/nc=10/names
    images/ labels/
  gan_runs/         # StyleGAN2-ADA 训练 run(00000=100kimg, 00001=800kimg)
  reports/
    COMPARISON.md   # 增广前后 YOLO 训练对比报告
    sheets/         # contact sheet + 指标图 + 位置分布图
```

## 复现链路

```bash
# 1. 提取裁剪(可复用脚本, --cls 指定类)
python aug/extract_crops.py --labels <labels/train> --images <images/train> \
    --out crops --cls 9 --size 128 --pad 1.6

# 2. 打包
python dataset_tool.py --source crops --dest gan_dataset.zip

# 3. 训练 GAN(128px, ADA, 800 kimg)
python train.py --outdir gan_runs --data gan_dataset.zip --gpus 1 \
    --metrics none --kimg 800 --snap 25

# 4. 生成
python generate.py --network gan_runs/.../network-snapshot-000800.pkl \
    --seeds 0-999 --outdir gen_800k --trunc 1

# 5. 回贴(按真实位置分布 + 避重叠 + 尺寸过滤)
python aug/paste_back.py --crops gen_800k --backgrounds <images/train> \
    --pos-labels <labels/train> --out pasted_full --cls 9

# 6. 组数据集(复制原 train + 贴入)
#    见 data_aug.yaml(已指向本实验 dataset/)

# 7. YOLO 训练对比(detect_model 项目)
#    scripts/train_yolo26n_metal_bucket_20260821.py
#    scripts/val_compare_metal_bucket_20260821.py
```

## 结果摘要

metal bucket mAP50 **+3.81%**(0.9317→0.9698),平均 mAP50-95 +1.10%。详见 `reports/COMPARISON.md`。

## 命名约定

- 实验标签 `<类名>_<YYYYMMDD>`(如 `metal_bucket_20260821`)。
- 可复用脚本(参数化,跨类通用)放 `aug/`;实验专属数据/产物/文档放 `experiments/<类名>_<日期>/`。
- 扩新类: 复制本目录结构,换 `--cls`、改日期标签即可。
