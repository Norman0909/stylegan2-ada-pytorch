# StyleGAN2-ADA 训练指标观察指南

## 一、生成指标图

```bash
# 在仓库根目录执行（用 base python，含 matplotlib）
python aug/plot_metrics.py --run runs/<run目录> --out <输出png>
```

读 `runs/<run目录>/stats.jsonl`（每行一条 JSON），画 6 张子图。训练中可反复跑，实时刷新。

---

## 二、六张子图怎么看

### 1. Loss（G/loss vs D/loss）
- `G/loss = softplus(-fake_logits)`，越低 = 生成图越能骗过 D。
- `D/loss = D_real + D_fake` 两部分。
- **正常**：两者小幅振荡，无长期发散。GAN 无"收敛到固定值"，别指望单调下降。
- **异常**：某一侧长期单调涨且不回落 → 训练失衡。

### 2. D logits（scores real vs fake）
- `real > 0`、`fake < 0` = D 判别正确，训练健康。
- **异常**：
  - 两者同号（都正或都负）→ D 已混淆，训练失效。
  - `real` 冲到很大正值 → D 过度自信，可能过拟合。

### 3. D logit 符号（signs，ADA 驱动）
- `signs/real` = 真图 logit 符号均值，**核心指标**。
- 目标值 `ada_target = 0.6`（图中红线）。
  - `signs/real > 0.6` → D 过拟合真图 → ADA 增强概率 p 上升（自动抗过拟合）。
  - `signs/real < 0.6` → D 未过拟合 → p 下降。
- `signs/fake` 应对称在负区间（越负 = D 越容易识破假图）。

### 4. ADA 增强概率 p（Progress/augment）
- `p ∈ [0, 1]`，0 = 不增强，1 = 满强度增强。
- **含义**：p 越高，说明 D 过拟合越严重、数据越"少而难"，ADA 自动加增强扛住。
- 我们的 metal bucket 只有 390 张，p 若长期贴近 1，说明数据确实太少。

### 5. R1 penalty（Loss/r1_penalty，对数轴）
- R1 梯度惩罚，约束 D 平滑、防过拟合、稳训练。
- **正常**：有限、不 NaN，量级 0.1~10 均常见。
- **异常**：爆到非常大（>1e4）或 NaN → 训练不稳，考虑降学习率 / 降 batch / 关 R1（`--gamma=0`）。

### 6. Path-length penalty（Loss/pl_penalty，对数轴）
- 约束 G 潜在空间平滑。越小越好。
- **异常**：长期不降或持续涨 → G 训练有问题。

---

## 三、健康判据速查

| 现象 | 含义 | 处理 |
|---|---|---|
| 任何指标出现 `NaN` | 训练崩溃 | 降 `--batch`、降学习率，或 `--fp32` |
| `signs/real` 长期 > 0.6 | D 过拟合 | 正常，ADA 会自动加增强 |
| `augment p` 长期贴近 1 | 数据太少 | 增数据，或接受较慢收敛 |
| `real`/`fake` logits 同号 | D 混淆 | 训练失效，检查数据/配置 |
| Loss 不降 + 生成图几乎相同 | mode collapse | 看 fakes 网格确认，加练或换方案 |
| `r1_penalty` 爆大 | 训练不稳 | 降 batch / 降学习率 / `--gamma=0` |

---

## 四、指标与生成质量的关系

**指标只反映训练动态，不直接等于生成质量。** 判断"生成图好不好"，必须配合：

1. `runs/<run>/fakes<kimg>.png` —— 训练循环定期导出的生成网格。
2. `runs/<run>/reals.png` —— 真实样本网格（对照）。
3. `python aug/inspect_gen.py --dir <生成目录> --out <sheet>` —— 生成图 contact sheet + 多样性统计。

**典型翻车案例（本项目 metal bucket）**：指标全部健康（loss 有限、无 NaN、ADA 正常），但生成图比真实更亮、物体占满整帧（`inspect_gen.py` 里 blob 尺寸 127×127 vs 真实 84×37）。说明 GAN 学到的分布跑偏。**必须看 fakes 网格 + 数值对比，不能只看 loss。**

---

## 五、字段速查（stats.jsonl）

| 字段 | 含义 |
|---|---|
| `Loss/G/loss` `Loss/D/loss` | 生成器 / 判别器损失 |
| `Loss/scores/real` `Loss/scores/fake` | D 对真 / 假图 logits |
| `Loss/signs/real` `Loss/signs/fake` | logits 符号均值（ADA 驱动） |
| `Loss/r1_penalty` | R1 梯度惩罚 |
| `Loss/pl_penalty` | path length 正则 |
| `Progress/kimg` | 训练进度（千张真图） |
| `Progress/augment` | ADA 增强概率 p |
| `Timing/sec_per_kimg` | 每 kimg 耗时（速度） |
| `Resources/peak_gpu_mem_gb` | GPU 峰值显存 |
