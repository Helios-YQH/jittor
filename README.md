# 点云降噪赛题 Baseline

## 环境安装
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

# 安装依赖
python -m pip install -r requirements.txt
pip install point-cloud-utils  # 可选，用于精确 P2S 评测
```

## 数据准备

1. 将训练数据 `dataset_train.tar.gz` 解压到本目录下：
   ```bash
   tar xzf dataset_train.tar.gz
   ```
   解压后目录结构：
   ```
   dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
   ```

2. 将测试数据 `dataset_test_noisy.zip` 解压到本目录下：
   ```bash
   unzip dataset_test_noisy.zip
   ```
   解压后目录结构：
   ```
   dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
   ```

3. 确保 `datalist/` 目录下存在训练/验证/测试的文件列表：
   ```
   datalist/train.txt
   datalist/validate.txt
   datalist/test.txt
   ```

## 快速验证（推荐首次运行）
使用小规模数据快速验证模型架构和代码是否正确，约 5 分钟完成：
```bash
python run.py --task configs/task/quick_train.yaml
```
该命令使用 200 个训练样本、4 的 batch size、5 个 epoch。正常运行时每个 epoch 的 loss 应逐步下降。若出现报错则说明环境或代码存在问题。

## 正式训练

### 单卡训练（快速调试）
```bash
python run.py --task configs/task/train_vm.yaml
```

### 分布式训练（推荐）
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" mpirun -np 6 python run.py --task configs/task/train_vm.yaml
```
- 框架自动完成数据分片和梯度同步，**无需修改代码**
- 每个 MPI 进程接收 `总 batch_size / 进程数` 个样本
- 仅 rank 0 保存 checkpoint 和输出日志，避免冲突

### 多次训练管理

训练权重保存在 `experiments/vm/` 目录下，每次运行时自动创建带时间戳的子目录，互不覆盖：

```
experiments/vm/
  checkpoint_20260518_143022/      ← 自动生成 run 子目录（东八区时间）
    checkpoint_0.pkl               ← 每个 epoch 保存一次
    checkpoint_1.pkl
    checkpoint_best.pkl            ← 验证 loss 最低时额外保存（最佳模型）
    training.log                   ← 仅包含本次运行的日志
    training_curve.png             ← 训练曲线图（每 epoch 更新）
  checkpoint_20260519_092115/      ← 另一次运行
    checkpoint_0.pkl
    ...
    training.log
```

**最佳模型机制（`checkpoint_best.pkl`）：** 每 epoch 结束时在验证集上计算 loss。如果当前 epoch 的 mean validation loss 为历史最低，则将模型保存为 `checkpoint_best.pkl`，并记录 `★` 标记。早停（early stopping）若开启（`patience: 10`），连续 `patience` 个 epoch 无改善则自动终止训练。

如需自定义运行名称，在 `configs/task/train_vm.yaml` 中设置：
```yaml
trainer:
  run_name: my_experiment_1    # 留空则自动生成时间戳
  patience: 10                 # 可选：开启早停
```

### 断点续训

在 `configs/task/train_vm.yaml` 中指定 `load_ckpt` 即可从之前的 checkpoint 继续训练：

```yaml
load_ckpt: experiments/vm/checkpoint_20260518_143022/checkpoint_best.pkl
```

- 留空（或注释掉）则从头训练，模型结构从 config 随机初始化
- 指定路径后：先根据 config 构建模型结构（保证子模块引用正确），再 `model.load()` 载入权重
- optimizer 状态**不保留**（学习率等从 config 重新初始化）
- 支持分布式训练、early-stopping、可视化等全部特性

## 自测评（推荐）
在提交前，使用自测评脚本评估模型在预留验证集上的降噪质量：
```bash
python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 10000
```
该脚本在验证集网格上采样 → 加噪 → 完整推理 → 计算 CD + P2S 得分，可帮助选择最佳 checkpoint。输出示例：

```
============================================================
  Self-Evaluation Results
============================================================
  Samples evaluated: 10
  Errors:            0
  Noise std:         0.0125
------------------------------------------------------------
  CD  (pred/noisy):  0.000668 / 0.000548
  Mean CD score:     1.13 / 100
  P2S (pred/noisy):  0.000073 / 0.000221
  Mean P2S score:    66.50 / 100
  Final score:       33.81 / 100
============================================================
```

- `num_samples` 控制每网格采样点数（默认 10000，越大越准但显存消耗高，24GB 卡建议不超过 20000）
- 训练早期 CD 提升慢于 P2S（先学表面法向，后学切向分布），属于正常现象

自测评与训练的 checkpoint 复用同一套加载机制：
- 在 `train_vm.yaml` 中设置 `load_ckpt` 指定要评估的权重路径
- 留空则自动选择 `experiments/vm/` 下最新 run 子目录中的 `checkpoint_best.pkl`

## 推理（生成提交文件）
修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 为你的最佳权重路径（含 run 子目录），例如：
```yaml
load_ckpt: experiments/vm/checkpoint_20260518_143022/checkpoint_best.pkl
```
然后运行：
```bash
python run.py --task configs/task/predict_vm.yaml
```
降噪结果保存在 `results/` 目录下。

## 打包提交
输出目录结构：
```
results/shapenet/<synset_id>/<model_id>/denoised.npy
```
打包命令：
```bash
cd results
zip -r ../result.zip shapenet/
```

## 提交格式
每个测试样本一个 `denoised.npy`，目录结构与测试集一致，打包为 `result.zip`：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

## 本地评测（需要 GT 数据，仅组委会持有）
```bash
python evaluate.py \
    --pred_dir ./results \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8
```
