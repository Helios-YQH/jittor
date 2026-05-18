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
```bash
python run.py --task configs/task/train_vm.yaml
```
训练权重保存在 `experiments/vm/` 目录下，每个 epoch 保存一次，并在验证 loss 最低时额外保存 `checkpoint_best.pkl`。

## 自测评（推荐）
在提交前，使用自测评脚本评估模型在预留验证集上的降噪质量：
```bash
python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 50000
```
该脚本在验证集网格上采样 → 加噪 → 完整推理 → 计算 CD + P2S 得分，可帮助选择最佳 checkpoint。

## 推理（生成提交文件）
修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 为你的最佳权重路径，然后运行：
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
