# StraightPCF 优化规划

基于论文 `StraightPCF: Straight Point Cloud Filtering` (2024) 的完整实现与改进方案。

## 当前项目与论文关键差异

| # | 问题 | 论文 | 当前代码 | 优先级 |
|---|------|------|---------|--------|
| 1 | **DistanceModule 死代码** | Eq.(15): d_φ/T * v 缩放步长 | (1/T) * v，无距离缩放 | **P0** ✅ |
| 2 | **Encoder 训练/推理不匹配** | 同分布输入提取特征 | 训练: pc_mix / 推理: pc_noisy | **P1** |
| 3 | **缺轨迹校正损失** | Eq.(10) λ₁ 校正中间点偏离 | 仅 loss_vm1 + loss_vm2 | **P2** |
| 4 | **未使用高噪声变体** | X_0 = Y + σ_H ξ (σ_H=2%) | X_0 = Y + actual_noise | **P2** |
| 5 | **Laplace vs Gaussian 噪声** | 论文全用 Gaussian | 比赛用 Laplace | **P3** |

---

## P0: 激活 DistanceModule ✅ (optimize-v1)

**状态**: 已实现，待训练验证

### 修改内容

**`src/model/distance_module.py`**:
- `execute()` 改为 `Max(head(encoder(features))) → Sigmoid`，输出每 batch 一个标量

**`src/model/coupled_vm.py`**:
- `deterministic_euler_step()` : 从初始状态特征计算 d_φ，缩放所有速度步
- `training_step()` : 新增 DistanceModule 损失
  - X_0 = pc_clean + Laplace(0, 0.02) 高噪声变体
  - X_t0 = (1-t)·X_0 + t·pc_clean, t~U(0,1)
  - target = 1-t (= 相对距离比)

### 注意事项
- 旧 checkpoint 不含 distance_module 权重（随机 init）
- 前几个 epoch d_φ 尚未收敛，推理结果可能波动
- 训练 1-2 epoch 后 d_φ 应迅速学习预测 (1-t)

---

## P1: 修复 Encoder 训练/推理输入不匹配

**问题**: 训练时 encoder 以 `pc_mix` (clean-noise 插值) 为输入，推理时以纯 `pc_noisy` 为输入。encoder 从未见过纯噪声数据。

**修复**: `VelocityModule.get_supervised_loss()` 中:
```python
# 修改前
feat = self.encoder(pc_mix)

# 修改后
feat = self.encoder(pc_noisy)
```

`pc_mix` 保留用于 decoder target 计算 (grad_dir_t_target = pc_clean - pc_noisy)

**预期影响**: 推理时 encoder 特征质量提升，CD 和 P2S 指标改善，尤其是 CD。

---

## P2: 补全 CoupledVM 训练 + 高噪声变体

### P2a: 轨迹校正损失 (论文 Eq.10)

当前 `CoupledVelocityModule.training_step`:
```python
loss = loss_vm1 + loss_vm2
```

论文的 `L_B`:
```
L_B = E_t[Σ||v_θ^k - δ(X₁,X₀)||² + λ₁Σ||δ(X̄_tk+1, X_tk+1)||²]
```
其中 λ₁=10，第二项鼓励中间过滤点 X̄_tk+1 接近插值点 X_tk+1。

**修复**: 添加校正项
```python
loss_correct = ||X_t1_pred - X_t1_interp||²
loss = loss_vm1 + loss_vm2 + λ₁ * loss_correct
```

### P2b: 高噪声变体训练 (论文 Eq.7)

论文的 velocity field target 是 `δ(X₁, X₀) = X₁ - X₀` 其中 X₀ 是 σ_H=2% 的高噪声变体。

当前代码 target 是 `pc_clean - pc_noisy`，其中 pc_noisy 是实际训练噪声 (0.5%-2%)。

**修复**: 使用 X₀ = pc_clean + Laplace(0, 0.02) 作为起点。

---

## P3: Laplace 噪声适配

**问题**: 论文训练和测试均用 Gaussian 噪声。比赛用 Laplace 噪声（tail 更重）。

**影响**:
- Laplace outliers 比 Gaussian 更多，模型可能不适应
- 插值路径 X_t = (1-t)X₀ + tX₁ 在 Laplace 噪声下仍为线性，velocity field 理论不受影响
- 但 encoder 在 Laplace 噪声点云上的特征提取能力可能下降

**缓解**:
1. 确保训练数据使用 Laplace 噪声 (当前已做到)
2. X₀ 高噪声变体也使用 Laplace 分布
3. 可选: 混合 Gaussian+Laplace 训练提高泛化

---

## P4 (远期): 架构与训练策略优化

### 4a. 模型容量调整
- `feat_embedding_dim`: 256 → 512 (提升特征表达能力)
- `num_train_points`: 128 → 256 (更多点级信息)
- 需要监控 GPU 内存

### 4b. 训练超参
- 延长 epoch: 100 → 300+ (CD 需要更多迭代)
- LR schedule: 推迟首次衰减 (epoch 60 而非 30)
- 增加 patch 数量: num_patches 8 → 12 (更多训练样本)

### 4c. 损失函数增强
- 引入 CD 辅助损失（采样固定点数计算 CD，加入总损失）
- 点分布正则化（防止聚类）

### 4d. 其他论文参考
- **P2P-Bridge** (diffusion bridges for denoising): 扩散桥接，可替代当前 Euler 步进
- **ScoreDenoise**: 分数匹配，可辅助 velocity field 学习
- **IterativePFN**: 内化迭代，可参考其 AGT 机制

---

## 训练迁移策略

从 `experiments/vm/checkpoint_20260519_084730/checkpoint_best.pkl` 出发：

1. **Phase 1** (P0, optimize-v1): 冻结 VM1/VM2，训练 DistanceModule 1-2 epoch
2. **Phase 2** (P0, optimize-v1): 解冻，全部参数微调 20+ epoch
3. **Phase 3** (P1): 修复 encoder 输入，继续训练
4. **Phase 4** (P2): 添加轨迹校正，继续训练
5. 每阶段保存独立 checkpoint 用于对比自测评
