# 热身赛：ModelNet40 点云分类 (PCT)

计图挑战赛赛道二热身赛 — 基于 Jittor 框架的 PCT (Point Cloud Transformer) 模型，完成 ModelNet40 三维形状分类任务。

> 技术报告（英文，含方法、工程细节与失败复盘）：[`../report/tech_report.pdf`](../report/tech_report.pdf)

## 赛题简介

- **任务**：输入 2048 个三维点，预测其所属类别（40 类）
- **训练集**：9843 个样本
- **测试集**：2468 个样本
- **通过线**：测试集准确率 ≥ 80%
- **框架**：Jittor（计图）
- **最终名次**：第 12 名

## 环境安装

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
pip install jittor numpy matplotlib
```

## 数据准备

下载 ModelNet40 点云数据，解压至 `warmup/data/` 目录：

```
warmup/data/
  train_points.npy    # (9843, 2048, 3)
  train_labels.npy    # (9843,)
  test_points.npy     # (2468, 2048, 3)
  categories.txt      # 40 个类别名称
```

## 快速开始

```bash
cd warmup
python train.py --data_dir ./data --epochs 300 --batch_size 32 --pct_full
```

`--pct_full` 是必须的：不加这个参数跑的是 SPCT 变体，而不是下面描述的最终模型。训练完成后生成 `result.json`，打包提交：

```bash
zip result.zip result.json
```

## 模型架构

最终版本是 PCT 论文 Figure 2b 的完整架构 (Point_Transformer2)：

```
Input: (B, 3, 1024)
  → Conv(3→64) + BN + ReLU
  → Conv(64→64) + BN + ReLU
  → SG-1: FPS(1024→512) + k-NN(k=32) + Local_op(128→128)
  → SG-2: FPS(512→256) + k-NN(k=32) + Local_op(256→256)
  → 4× SA_Layer (l1-Norm Offset-Attention + position encoding)
  → Concat(skip + SA outputs) → Conv(1280→1024)
  → Max Pool + Classifier(1024→512→256→40)
```

- 参数量：约 2.88M
- 与参考实现的一处差异：4 层 SA 各带独立的位置编码投影（`pos_conv1..4`），论文中共用一个。没有做消融验证这一改动的效果

## 训练策略

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW (weight_decay=1e-4) |
| 学习率 | 0.01, WarmupCosineAnnealing (10 epoch warmup, min=1e-6) |
| Batch Size | 32 |
| Epochs | 300 |
| 数据增强 | 随机缩放(0.67-1.5x) + 平移(±0.2) + 点丢弃(10-40%) + 高斯抖动(σ=0.01) |
| 正则化 | Label Smoothing (0.1) + Dropout (0.5) + EMA |
| 验证策略 | 90/10 训练/验证固定划分 (seed=42) |

EMA 需要显式传参（`--ema_decay 0.999`），命令行默认值是 0.0，即默认不启用。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据目录 |
| `--n_points` | `1024` | 输入点数 |
| `--batch_size` | `32` | 批次大小 |
| `--epochs` | `300` | 训练轮数 |
| `--lr` | `0.01` | 初始学习率 |
| `--pct_full` | False | 使用完整 PCT；不开则使用 SPCT |
| `--optimizer` | `adamw` | 优化器 (sgd/adamw/adam) |
| `--label_smooth` | `0.1` | Label Smoothing 系数 |
| `--ema_decay` | `0.0` | EMA 衰减系数，不传则不启用 |
| `--pretrained` | None | SPCT 预训练权重路径 |

## 文件说明

| 文件 | 用途 |
|------|------|
| `train.py` | 主训练和推理脚本（最终版本） |
| `rf_pct.py` | PCT 基础模块（SA_Layer, Local_op, Point_Transformer2, sample_and_group） |
| `rf_ops.py` | 底层算子（FPS, KNN, Ball Query, index_points），含 CUDA 内核 |
| `show.py` | 数据增强可视化调试工具 |

## 改进历程

从官方 baseline 到最终版本：

1. **官方 baseline** — 简单 PCT (4×SA, 无 Neighbor Embedding)，仅做绕 Y 轴旋转增强，SGD + CosineAnnealing，准确率约 80%。
2. **几何编码实验** — SA 层加入基于欧式距离的注意力偏置，配 LayerNorm + FFN。效果不如预期，弃用。
3. **层次化 PCT** — 引入 `rf_pct` 的采样分组下采样 + SA 注意力，增强策略扩展为缩放+平移+点丢弃+抖动，Adam + WarmupCosine。
4. **最终版本** — 对齐论文完整架构（Point_Transformer2），引入 Label Smoothing + EMA + AdamW，支持 SPCT → 完整 PCT 的权重迁移。

第 1–3 步的代码文件在仓库中已不存在（当时的 `WarmUp/` 目录未纳入版本控制），只能从 git 记录和上面的描述回溯。各版本的准确率没有留存记录，最终名次是仅存的评测结果。

## 来源与许可

- `rf_ops.py` 中的 `index_points`、`square_distance`、`PointNetFeaturePropagation`、ball query 与 KNN 的 CUDA 实现移植并改编自 PointNet++ 及其公开实现（Charles R. Qi 等，MIT 许可），为在 Jittor 下运行做了修改。
- 模型结构遵循 PCT 论文（Guo 等，*Point Cloud Transformer*, Computational Visual Media 2021）。
- 本仓库其余代码采用 MIT 许可，见 [`../LICENSE`](../LICENSE)。
