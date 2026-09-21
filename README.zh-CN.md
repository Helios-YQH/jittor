# 计图挑战赛赛道二 — 点云去噪与点云分类

第六届计图人工智能挑战赛赛道二的个人参赛作品，基于 [Jittor（计图）](https://github.com/Jittor/jittor) 实现：

- **正式赛 —— 点云去噪**：复现 **StraightPCF**（CVPR 2024），预测每个点的位移向量，把含噪的 ShapeNet 点云推回物体表面。
- **热身赛 —— 三维形状分类**：基于 **PCT**（Point Cloud Transformer）的 ModelNet40 分类模型。

**技术报告：[`report/tech_report.pdf`](report/tech_report.pdf)** —— 方案实现、Jittor 工程细节、实测结果，以及一篇诚实的失败复盘（"哪些没做成、为什么"）。([LaTeX 源码](report/tech_report.tex) · [English README](README.md))

## 结果一览

| 赛题 | 模型 | 结果 |
|---|---|---|
| 正式赛（去噪） | StraightPCF 复现 —— 耦合速度模块，约 0.7M 参数 | 竞赛最好成绩 **67.44/100**（CD 子分 51.9，P2S 子分 82.9）；相对含噪输入，CD 下降 51%、P2S 下降 67%；全国前 100 名 |
| 热身赛（分类） | PCT —— 2 个下采样分组模块 + 4 层 Offset-Attention，约 2.9M 参数 | 通过热身赛（测试集准确率 ≥ 80% 通过线） |

以上数字均在训练网格的留出集上按竞赛指标（Chamfer 距离 + 点到面距离，逐样本计分）测得。本复现**没有**达到 StraightPCF 论文报告的精度，报告中复盘一节给出了我们的分析。

## 仓库结构

```
├── denoise/                 # 正式赛：点云去噪
│   ├── run.py               #   训练 / 推理 / 调试入口
│   ├── self_eval.py         #   留出集自评测（CD + P2S，竞赛计分）
│   ├── evaluate.py          #   官方风格评测脚本
│   ├── profile.py           #   数据与计算耗时分析、batch size 探测
│   ├── vis_denoising.py     #   逐样本去噪诊断可视化
│   ├── configs/             #   YAML 配置：task / data / model / system / transform
│   └── src/                 #   数据管线、模型、训练循环
│       ├── data/            #     网格采样、归一化、噪声、patch 构造
│       ├── model/           #     EdgeConv 编码器、速度模块、距离模块
│       └── system/          #     训练器（MPI、断点、调度）、结果写出
├── warmup/                  # 热身赛：ModelNet40 分类（PCT）
│   ├── train.py             #   训练 + 推理
│   ├── rf_pct.py            #   PCT 基础模块（Offset-Attention、采样分组）
│   └── rf_ops.py            #   底层算子（FPS、KNN、Ball Query）
├── report/                  # 技术报告：LaTeX 源码、PDF、图、训练日志
│   ├── tech_report.pdf
│   ├── tech_report.tex
│   └── make_figures.py      #   由训练日志重绘报告中的图
└── README.zh-CN.md          # 本文件
```

## 快速开始

环境（详见 `denoise/README.md`）：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
pip install jittor numpy trimesh scipy omegaconf matplotlib
pip install point-cloud-utils   # 可选，用于精确 P2S 评测
```

去噪（路径相对于 `denoise/`）：

```bash
cd denoise

# 单卡训练（竞赛所用配置）
python run.py --task configs/task/train_vm.yaml

# MPI 多卡分布式训练
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" mpirun -np 6 python run.py --task configs/task/train_vm.yaml

# 推理（先在配置中指定 load_ckpt）
python run.py --task configs/task/predict_vm.yaml

# 留出集自评测（CD + P2S，竞赛计分）
python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 10000
```

论文的分阶段训练有独立配置：`configs/task/train_vm_stage{1..4}.yaml`。基于 score 的变体（ScoreDenoise 风格）在 `src/model/score_vm.py`，配置为 `train_score.yaml` / `predict_score.yaml`；该路线未训练到可用状态。

分类（热身赛）：

```bash
cd warmup
python train.py --data_dir ./data --epochs 300 --batch_size 32
```

## 数据

数据集（训练用 ShapeNet 网格、测试用含噪点云）由赛事主办方提供，本仓库**不**包含。代码默认数据放在 `denoise/dataset_train/` 与 `denoise/dataset_test_noisy/`，划分列表在 `denoise/datalist/`。目录格式见 `denoise/README.md`。

## 文档

| 文档 | 语言 | 内容 |
|---|---|---|
| [`report/tech_report.pdf`](report/tech_report.pdf) | 英文 | 方法、Jittor 工程实现、结果、失败复盘 |
| [`denoise/README.md`](denoise/README.md) | 中文 | 去噪：完整用法、配置说明、打包提交、常见问题 |
| [`warmup/README.md`](warmup/README.md) | 中文 | 分类：模型架构、训练策略、改进历程 |
| [`denoise/ANALYSIS.md`](denoise/ANALYSIS.md) | 中文 | 分数退化分析与分阶段复现计划的工作笔记 |

## 状态

比赛已于 2026 年结束。仓库作为记录保留：代码保持原样归档，`report/tech_report.pdf` 记录了什么做成了、什么没做成。

## 参考文献

- StraightPCF: *Straight Point Cloud Filtering*, CVPR 2024 —— 复现的方法。
- ScoreDenoise: *Score-Based Point Cloud Denoising*, ICCV 2021 —— `src/model/score_vm.py` 中的 score 变体。
- PCT: *Point Cloud Transformer*, Computational Visual Media 2021 —— 热身赛分类器。
