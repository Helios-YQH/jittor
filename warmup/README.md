# 热身赛：ModelNet40 点云分类 (PCT)

计图挑战赛赛道二热身赛 — 基于 Jittor 框架的 PCT (Point Cloud Transformer) 模型，完成 ModelNet40 三维形状分类任务。

## 赛题简介

- **任务**：输入 2048 个三维点，预测其所属类别（40 类）
- **训练集**：9843 个样本
- **测试集**：2468 个样本
- **通过线**：测试集准确率 ≥ 80%
- **框架**：Jittor（计图）

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
python train.py --data_dir ./data --epochs 300 --batch_size 32
```

训练完成后自动在 `warmup/` 目录下生成 `result.json`，打包提交：

```bash
zip result.zip result.json
```

## 模型架构

最终版本使用 Point_Transformer2（PCT 论文 Figure 2b 完整架构）：

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

- 参数量：~2.88M
- 核心改进：Offset-Attention + Neighbor Embedding + 多尺度特征融合

## 训练策略

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW (weight_decay=1e-4) |
| 学习率 | 0.01, WarmupCosineAnnealing (10 epoch warmup, min=1e-6) |
| Batch Size | 32 |
| Epochs | 300 |
| 数据增强 | 随机缩放(0.67-1.5x) + 平移(±0.2) + 点丢弃(10-40%) + 高斯抖动(σ=0.01) |
| 正则化 | Label Smoothing (0.1) + EMA (decay=0.999) + Dropout (0.5) |
| 验证策略 | 90/10 训练/验证固定划分 (seed=42) |

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据目录 |
| `--n_points` | `1024` | 输入点数 |
| `--batch_size` | `32` | 批次大小 |
| `--epochs` | `300` | 训练轮数 |
| `--lr` | `0.01` | 初始学习率 |
| `--pct_full` | False | 使用 Full PCT（Point_Transformer2）；否则使用 SPCT |
| `--optimizer` | `adamw` | 优化器 (sgd/adamw/adam) |
| `--label_smooth` | `0.1` | Label Smoothing 系数 |
| `--ema_decay` | `0.0` | EMA 衰减系数 |
| `--pretrained` | None | SPCT 预训练权重路径（用于迁移学习） |

## 文件说明

| 文件 | 用途 |
|------|------|
| `train.py` | 主训练和推理脚本（最终版本） |
| `rf_pct.py` | PCT 基础模块（SA_Layer, Local_op, Point_Transformer2, sample_and_group） |
| `rf_ops.py` | 底层 CUDA 算子（FPS, KNN, Ball Query, index_points） |
| `show.py` | 数据增强可视化调试工具 |

## 改进历程

从官方 Baseline 到最终版本，主要经历了以下尝试：

1. **baseline.py** — 官方基础代码。简单 PCT (4×SA, 无 Neighbor Embedding)，仅做绕 Y 轴旋转增强，SGD + CosineAnnealing。准确率约 80%。

2. **pct.py** — 第一轮改进。引入 Hierarchical_PCT（基于 rf_pct 的 SG 下采样 + SA 注意力），增强策略扩展为缩放+平移+点丢弃+抖动，Adam + WarmupCosine，新增训练/验证集划分。

3. **pct0429.py** — 实验性改进。SA 层加入几何编码（基于欧式距离的注意力偏置）、LayerNorm + FFN 结构。经实验效果不如预期，未采用。

4. **train.py（最终）** — 对齐论文完整架构。完整实现 Point_Transformer2（SG×2 + 4×SA + 独立位置编码 + Skip Connection），引入 Label Smoothing + EMA + AdamW 等 SOTA 训练策略。支持 SPCT → Full PCT 权重迁移。
