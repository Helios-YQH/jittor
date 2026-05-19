# 去噪模型问题诊断

**检查点**: `checkpoint_best.pkl` (epoch 89, 100 轮训练)  
**自测评结果**:
```
CD  (pred/noisy):  0.000573 / 0.000548  →  CD score:  8.24 / 100
P2S (pred/noisy):  0.000086 / 0.000221  →  P2S score: 60.58 / 100
Final score: 34.41 / 100
```

**核心症状**: P2S 提升 61%，但 CD 反比 noisy 差 4.6%。模型学会了把点往表面拉（法向），但没学会正确的切向分布。

---

## 问题 1：Encoder 训练/推理输入分布不匹配

**这是最关键的 bug。**

### 现象

- **训练时** (`src/model/vm.py:54`): encoder 的输入是 `pc_mix = t*clean + (1-t)*noisy`，即 clean-noise 插值
- **推理时** (`src/model/vm.py:84`): encoder 的输入是纯 noisy 或去噪中的纯点云

encoder 在整个训练期间从未见过纯 noisy 输入，只在插值数据上训练。

### 代码证据

```python
# vm.py:44 - get_supervised_loss
def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
    feat = self.encoder(pc_mix)  # ← 使用混合点云提取特征
    ...

# vm.py:84 - deterministic_euler_step  
def deterministic_euler_step(self, pcl_noisy, num_steps=3):
    feat = self.encoder(pcl_next)  # ← 使用纯去噪中的点云
    ...
```

### 影响

推理时 encoder 提取的特征质量远低于训练时，导致 decoder 的速度场预测偏差。这解释了为什么 val loss 在 0.16 时平台，但推理指标更差 — encoder 在训练时"作弊"看到了部分 clean 信息，推理时没有。

### 修复

将 `get_supervised_loss` 中 encoder 输入改为 `pc_noisy`：

```python
feat = self.encoder(pc_noisy)  # 替代 pc_mix
```

`pc_mix` 可保留用于构造损失（如继续使用混合点云的位移作为 target），但特征提取必须与推理时一致。

---

## 问题 2：CD 与 P2S 的收敛速度差异

### 现象

- P2S 提升快 (61%): 模型只需将点朝最近表面移动即可
- CD 反而退化 (+4.6%): 点被拉向表面时发生局部聚集，切向分布未学会

### 根因

DSM 损失是各向同性的逐点 L2 位移：
```
L = MSE(pred_direction, clean_position - noisy_position) / sigma
```

它不区分：
- 法向分量 (主导 P2S) — 信号强，容易学
- 切向分量 (主导 CD) — 信号弱，需更多迭代

在训练早期，模型预测的位移以法向为主，点向表面聚集 → NN 距离增大 → CD 退化。

### 趋势

这是 DSM 训练的典型过渡阶段。更多训练后切向分量会逐渐学成，但需要远超 100 个 epoch。

---

## 问题 3：Val Loss 收敛平台

### 现象

100 epoch 后 train loss (0.17) ≈ val loss (0.16)，差距极小。后 50 epoch 几乎无改进。

| 阶段 | Train Loss | Val Loss | 改进幅度 |
|------|-----------|----------|---------|
| Epoch 0-10 | 0.20 → 0.18 | 0.19 → 0.18 | 显著 |
| Epoch 10-50 | 0.18 → 0.17 | 0.18 → 0.17 | 缓慢 |
| Epoch 50-100 | 0.17 → 0.17 | 0.17 → 0.16 | 极微 |

### 可能原因

1. 模型容量不足: `feat_embedding_dim=256`, `decoder_hidden_dim=64` 可能不足以捕捉细粒度切向位移
2. Patch 训练视野有限: `patch_size=1000` 只能捕捉局部几何，不支持全局点分布学习
3. 优化器 LR 衰减过早: epoch 30 时 LR → 1e-5，此时 val loss 仍在改善，降低 LR 可能过早限制了学习
4. 问题 1 的 encoder 不匹配限制了 loss 下界

---

## 问题 4：训练数据流与 Patch 构建

### 数据增强流程

```
mesh → sample(32768点) → normalize → add_noise(0.005~0.020) 
     → augment_linear(旋转/缩放) → patch(patch_size=1000, num_patches=8)
```

### Patch 构建 (`src/data/augment.py:171-207`)

1. 从 noisy 点云随机选 8 个 seed point
2. 对每个 seed，在 noisy 点云中用 cKDTree 找最近的 1000 个点
3. 构造 `pc_mix = t*pc_clean + (1-t)*pc_noisy`，t ~ U(1e-8, 1.0)

### 潜在问题

- `pc_mix` 使用 noisy 的 KNN 邻域结构 (tree 建在 pc_noisy 上)，这意味着无论 t 取什么值，邻域关系始终保持 noisy 的
- 但 `pc_mix` 的点坐标是 clean-noisy 的线性插值，与 KNN 邻域不一致

---

## 总结：根因优先级

| 优先级 | 问题 | 预期影响 | 实施难度 |
|--------|------|---------|---------|
| **P0** | Encoder 训练/推理输入不匹配 | 高 — 限制模型上限 | 低 |
| **P1** | CD 需更多训练 (300+ epoch) | 中 — 切向分布需大量迭代 | 低 |
| **P2** | 模型容量/训练参数调优 | 中 — 可提升上限 | 中 |
| P3 | 训练数据流优化 | 低 — 微调效果 | 高 |

---

## 可视化诊断工具

运行 `vis_denoising.py` 生成对比图，用于暴露去噪效果问题：

```bash
python vis_denoising.py --task configs/task/train_vm.yaml --num_samples 3 --vis_points 2000
```

输出文件：
- `vis_output/sample_N_comparison.png`: 三列对比 (noisy / denoised / clean)，点按 NN 误差着色
- `vis_output/sample_N_displacement.png`: 预测位移向量及方向角误差

> 注意: 本地仅调试用，实际运行需在云端提交。
