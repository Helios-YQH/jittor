# 正式赛：基于深度学习的点云降噪 (StraightPCF)

计图挑战赛赛道二正式赛题 — 基于 Jittor 框架实现 StraightPCF 点云降噪模型，从含噪点云中恢复干净表面。

> 技术报告（英文，含方法、工程细节与失败复盘）：[`../report/tech_report.pdf`](../report/tech_report.pdf)
>
> **注意**：仓库里的代码是 7 月重构后的版本，与产生 67.44 分的那份提交**不是同一套管线**——提交版直接在真实含噪点云上训练、不使用 DistanceModule 与 coupling 项；7 月改为论文式的合成噪声训练后掉到 64.79。两者差异见技术报告第 3 节。

## 赛题简介

给定从三维物体表面采样并受噪声污染的点云，模型需要预测每个点的位移向量，将含噪点"推回"到真实物体表面附近，输出降噪后的点云。

- **输入**：含噪点云 `{p_i + n_i}`，Noise ~ Laplace(0, σ), σ ∈ [0.005, 0.020]（归一化至单位球）
- **输出**：降噪点云，点数与输入严格一致，float32, shape (N, 3)
- **训练集**：~20,000 个 ShapeNet 三维网格 (.obj)
- **测试集**：含噪点云 (.npy)
- **框架**：Jittor（计图）

## 方法

本方案基于 **StraightPCF** (CVPR 2024)：将点云去噪建模为最优传输问题，通过 VelocityModule 学习恒定速度场，使含噪点沿直线路径移动到干净表面。

### 架构

```
Input Noisy Patch
      ↓
FeatureExtraction (Dynamic EdgeConv × 3, KNN graph)
      ↓
┌─────────────────────────────────────┐
│  Coupled VelocityModule (K=2)       │
│  ┌──────────┐    ┌──────────┐       │
│  │   VM1    │ → │   VM2    │       │
│  │ velocity │    │ refined │       │
│  │  field   │    │ velocity │       │
│  └──────────┘    └──────────┘       │
│       ↓               ↓             │
│  DistanceModule(d_φ) — scales step  │
└─────────────────────────────────────┘
      ↓
Euler Integration (3 steps × K repeats)
      ↓
Denoised Output
```

- **FeatureExtraction**: 3 层 DynamicEdgeConv (k=16, feat_dim=256)
- **FeatureExtraction**: 3 层 DynamicEdgeConv (k=16)。提交版通道为 3→32→64→96→256，当前代码为 3→64→128→192→256
- **Decoder**: MLP (256 → 256 → 64 → 3)，输出三维位移向量
- **DistanceModule**: 小型 MLP (256 → 64 → 64 → 1)，按 patch 取 max 后过 sigmoid，输出距离标量 d_φ ∈ [0,1] 缩放步长。**提交版本没有使用它**（训练目标里没有该项，推理的 Euler 步进也已摘除）
- 参数量：提交版两个速度模块合计约 0.47M；当前代码（更宽的编码器 + 距离模块）约 0.71M

### 论文参考

> Dasith de Silva Edirimuni, Xuequan Lu, Gang Li, Lei Wei, Antonio Robles-Kelly, Hongdong Li. *StraightPCF: Straight Point Cloud Filtering.* CVPR 2024.

详见 `papers/StraightPCF.md` 和方法论文 `papers/Straight_Point_Cloud_Filtering.pdf`。

## 环境安装

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y
conda install -c conda-forge libgomp -y

cd denoise
pip install -r requirements.txt
pip install point-cloud-utils  # 可选，用于精确 P2S 评测
```

依赖：`jittor`, `numpy`, `trimesh`, `scipy`, `omegaconf` (+ 可选的 `point-cloud-utils`, `matplotlib`)

## 数据准备

### 1. 训练数据

```bash
tar xzf dataset_train.tar.gz
```

解压后目录结构：

```
dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

### 2. 测试数据

```bash
unzip dataset_test_noisy.zip
```

解压后目录结构：

```
dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

### 3. 数据列表

在 `datalist/` 目录下准备三个文件列表（每行一个相对路径）：

```
datalist/
  train.txt       # 训练样本路径列表
  validate.txt    # 验证样本路径列表
  test.txt        # 测试样本路径列表
```

## 快速验证

使用小规模数据快速验证模型架构和代码：

```bash
python run.py --task configs/task/quick_train.yaml
```

该命令使用 200 个训练样本、batch_size=4、5 个 epoch。正常运行时 loss 应逐步下降。

## 训练

### 单卡训练

```bash
python run.py --task configs/task/train_vm.yaml
```

### 多卡分布式训练（推荐）

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" mpirun -np 6 python run.py --task configs/task/train_vm.yaml
```

Jittor 自动处理数据分片、参数广播和梯度同步。

### 断点续训

在 `configs/task/train_vm.yaml` 中指定 `load_ckpt`：

```yaml
load_ckpt: experiments/vm/checkpoint_20260518_143022/checkpoint_latest.pkl
```

### 训练配置

| 配置项 | 值 | 文件 |
|--------|-----|------|
| 模型 | CoupledVelocityModule | `configs/model/vm.yaml` |
| Batch Size | 24 | `configs/data/train.yaml` |
| 优化器 | Adam (lr=1e-4) | `configs/task/train_vm.yaml` |
| 学习率调度 | cosine warm restart | `configs/task/train_vm.yaml` |
| Epochs | 100 | `configs/task/train_vm.yaml` |
| 采样点数 | 32768 | `configs/transform/vm.yaml` |
| Patch 大小/数量 | 1000 / 8 | `configs/transform/vm.yaml` |
| 噪声范围 | Laplace, σ ∈ [0.005, 0.020] | `configs/transform/vm.yaml` |

### 训练输出

```
experiments/vm/
  checkpoint_20260518_143022/   ← 自动生成带时间戳的子目录
    checkpoint_0.pkl            ← 每 epoch 保存一次
    checkpoint_latest.pkl        ← 每 epoch 覆盖，最新权重
    training.log                ← 训练日志
    training_curve.png          ← 训练曲线图
```

### Batch Size 参考

| 卡数 | 全局 batch_size | 每卡样本数 | 约显存 |
|------|----------------|-----------|--------|
| 1 | 12 | 12 | ~8GB |
| 1 | 24 | 24 | ~12GB |
| 6 | 72 | 12 | ~8GB/卡 |
| 6 | 144 | 24 | ~12GB/卡 |

## 推理

修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 指向最佳权重：

```yaml
load_ckpt: experiments/vm/checkpoint_20260518_143022/checkpoint_latest.pkl
```

运行推理：

```bash
python run.py --task configs/task/predict_vm.yaml
```

降噪结果保存在 `results/` 目录下。

## 打包提交

```bash
cd results
zip -r ../result.zip shapenet/
```

提交格式：

```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3), N 与输入严格一致
```

## 自测评

在提交前评估模型在预留验证集上的降噪质量：

```bash
python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 10000
```

输出示例：

```
============================================================
  Self-Evaluation Results
============================================================
  Samples evaluated: 10
  Noise std:         0.0125
------------------------------------------------------------
  CD  (pred/noisy):  0.000668 / 0.000548
  Mean CD score:     1.13 / 100
  P2S (pred/noisy):  0.000073 / 0.000221
  Mean P2S score:    66.50 / 100
  Final score:       33.81 / 100
============================================================
```

## 评测指标

### Chamfer Distance (CD)

$$CD(S_{pred},S_{gt})=\frac{1}{|S_{pred}|}\sum_{x \in S_{pred}} \min_{y \in S_{gt}} \|x-y\|_2^2 + \frac{1}{|S_{gt}|}\sum_{y \in S_{gt}} \min_{x \in S_{pred}} \|y-x\|_2^2$$

### Point-to-Surface Distance (P2S)

$$P2S(S_{pred},M)=\frac{1}{|S_{pred}|}\sum_{x \in S_{pred}} \min_{y \in M} \|x-y\|_2^2$$

### 百分制评分

$$cd\_score_i = clamp(100 \times (1-\frac{CD_{pred(i)}}{CD_{noisy(i)}}),\ 0,\ 100)$$

$$p2s\_score_i = clamp(100 \times (1-\frac{P2S_{pred(i)}}{P2S_{noisy(i)}}),\ 0,\ 100)$$

$$Score_{final} = \frac{1}{2} \times mean(cd\_score) + \frac{1}{2} \times mean(p2s\_score)$$

## 官方评测脚本

```bash
python evaluate.py \
    --pred_dir ./results \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8 \
    --verbose
```

## 可视化诊断

```bash
python vis_denoising.py --task configs/task/predict_vm.yaml --num_samples 3 --vis_points 2000
```

每样本生成 4 张诊断图，保存至 `vis_output/`：
- 3D 点云对比（含噪/降噪/干净）
- 位移向量分析（方向角误差）
- 误差分布直方图
- 位移质量分析（幅度 + 方向）

## 性能分析

```bash
python profile.py                    # 自动探测 batch_size
python profile.py --batch_size 12    # 手动指定
```

输出数据加载 vs GPU 计算耗时占比，帮助判断瓶颈。

## 项目结构

```
denoise/
  run.py                  # 主入口（训练/推理/调试）
  evaluate.py             # 官方评测脚本 (CD + P2S)
  self_eval.py            # 自测评脚本
  profile.py              # 性能分析
  vis_denoising.py        # 可视化诊断
  requirements.txt        # Python 依赖

  configs/
    task/                 # 任务配置 (train/predict/quick_train/debug/stage1-4/score)
    data/                 # 数据配置 (train/quick_train/predict)
    model/                # 模型配置 (vm/score)
    transform/            # 变换配置 (vm/predict/score)
    system/               # 系统配置 (vm/dummy)

  src/
    data/
      asset.py            # Asset 数据结构 + OBJ 导出
      augment.py          # 数据增强 (采样/归一化/加噪/旋转/patch)
      datapath.py         # 数据路径管理与延迟加载
      dataset.py          # Jittor Dataset + DataLoader
      spec.py             # ConfigSpec 基类
      transform.py        # Transform 增强管道
      utils.py            # 网格采样 / 重心插值 / 随机旋转
    model/
      spec.py             # ModelSpec 基类
      parse.py            # 模型工厂函数
      feature.py          # DynamicEdgeConv / FeatureExtraction / Decoder
      vm.py               # VelocityModule + FPS/KNN/patch 去噪
      coupled_vm.py       # CoupledVelocityModule (VM1+VM2+DistanceModule)
      distance_module.py  # DistanceModule (距离标量 d_φ)
      score_vm.py         # ScoreVelocityModule (ScoreDenoise 变体，未训练到可用)
    system/
      spec.py             # 训练循环 / 验证 / 推理 / 早停
      parse.py            # 系统工厂函数
      vm.py               # VMWriter (结果输出)
```

## 常见问题

### 训练中途崩溃 (Segfault)

```
Caught segfault at address 0x7f..., thread_name: '', flush log...
```

已内置 `jt.sync_all()` + `jt.gc()` 自动防护。如仍遇到问题，降低 batch_size。

### CD 分数低 / P2S 分数高

我们当时也停在这个状态，最终没有解决。事后复盘的结论是：把训练数据换成论文式的"干净点云 + 合成高斯噪声"之后，模型与比赛使用的 Laplace 噪声脱节——单独用高斯噪声训练的模块在比赛分布上得分 1.01/100。训练更久没有帮助：两个阶段的 loss 都在 5–40 个 epoch 内到平台。详见技术报告第 6 节。

### GPU 显存不足 (OOM)

- 降低 `configs/data/train.yaml` 中的 `batch_size`
- 使用多卡 MPI 分布式训练分摊 batch
- 减小 `configs/transform/vm.yaml` 中的 `patch_size` 或 `num_patches`

## 注意事项

- 比赛需使用 Jittor 框架
- 不得使用提供数据集之外的其他数据
- 所有团队需开源代码方视为有效成绩
- 每队伍每天最多提交 2 次
- 降噪点云点数必须与输入含噪点云严格一致

## 来源与许可

- 方法、损失函数与网络结构来自 StraightPCF（de Silva Edirimuni 等，*Straight Point Cloud Filtering*, CVPR 2024），编码器/解码器的结构参照其官方实现编写。
- 本仓库代码采用 MIT 许可，见 [`../LICENSE`](../LICENSE)。
