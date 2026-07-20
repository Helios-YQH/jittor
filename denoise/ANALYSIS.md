# StraightPCF 复现分析

## 当前状态 (2026-07-20)

| 指标 | May 最佳 | 当前最佳 | 变化 |
|---|---|---|---|
| Final Score | 67.44 | 64.79 | -2.65 |
| CD Score | 51.94 | 51.65 | -0.29 |
| P2S Score | 82.94 | 77.93 | **-5.01** |
| mean CD_pred | 0.000121 | 0.000120 | ≈ |
| mean CD_noisy | 0.000246 | 0.000246 | = |
| mean P2S_pred | 0.000064 | 0.000072 | ↑ (worse) |
| mean P2S_noisy | 0.000196 | 0.000196 | = |

退化集中在 P2S = 点无法回到表面。DistanceModule 在代码中没按论文阶段训练是最可能的根因。

## 根本原因分析

### P2S 退化 → DistanceModule 失效
- 论文 Stage 4: Backbone 永远冻结, 只训 DM, Term2 的 λ₂=200 通过 Euler 链"教会"DM 不是简单的 (1-t) 回归, 而是"多少步长能让点最终停在表面"
- 当前代码 Phase 2a: DM 训练 5 epoch 后 → Phase 2b 解冻 backbone → DM 输入分布剧变 → 之前 Term2 学到的步长策略作废
- 当前代码 Phase 2b 没有 Term2 (loss_dist_term2 = 0.0), DM 退化为纯 (1-t) 回归, 丧失了"收敛到表面"的引导

### CD 稳定 → 速度场本身尚可
- VM1/VM2 的速度方向大致正确 (CD 没退化)
- 但没有 DM 正确的步长缩放, 点不是正好停在表面, 所以 P2S 差

## 偏离论文的关键问题

1. **噪声类型**: 论文全程 Gaussian σ_H=2%, 通过 X_t = (1-t)X_0 + tX_1 的数学框架泛化。当前代码 Phase 2 用 mixed 噪声, 破坏了这个假设
2. **训练阶段**: 4 阶段合并为 2 阶段, Stage 1/2 的单 VM pretrain 被跳过
3. **DM 训练**: Backbone 必须永远冻结 (论文明确说 "keep weights fixed"), 当前代码解冻了
4. **推理重复**: 论文单次 Euler, 当前代码 2-4 轮重复 patch_based_denoise — 掩盖速度场不够直

## 严格复现计划

## Epoch 收敛速度观察

Phase 1 (CoupledVM 混训): Epoch 0 loss 0.0556 → 1 降到 0.0066 (下降 89%) → 5 降到 0.0012 → 之后 240 epochs 平台 ~0.0010
Phase 2 (DM 训练): Epoch 0 0.0310 → 5 降到 0.0014 → 之后 25 epochs 平台 ~0.0014

结论: 纯 VM 训练 5 epoch 就收敛, 延长时间无效。每个 epoch ~930s。

## 严格复现计划

### Stage 1: Pretrain VM1 (15 epochs, ~3.9h)
- 单 VelocityModule, Eq.(7): L = ||v_θ(X_t) - (X_1 - X_0)||²
- 纯 Gaussian σ_H=2%, 无 DM, 无 VM2
- num_train_points = 所有 patch 点 (1000)
- Config: `configs/task/train_vm_stage1.yaml`
- Model: `configs/model/vm_stage1.yaml` (__target__: VelocityModule)
- Transform: `configs/transform/vm_stage1.yaml` (noise_type: gaussian)
- 训练结束后: `python self_eval.py --task configs/task/train_vm_stage1.yaml` 评估效果
- 输出: save checkpoint_0.pkl ~ checkpoint_14.pkl, 选 self_eval 最佳作为 Stage 1 权重

### Stage 2: Pretrain VM2 (~15 epochs)
- 同 Stage 1, 独立权重, 仅改 seed + run_name

### Stage 3: Coupled Finetune (~15 epochs)
- 加载 Stage 1+2 权重到 CoupledVelocityModule, Eq.(10) coupling loss (λ₁=10)
- DM 随机初始化 + 冻结

### Stage 4: DM Training (~20 epochs)
- Backbone 永远冻结
- Term1: ||d_φ - (1-t)||²
- Term2: λ₂=200 · ||X̄ - X₁||² (通过 Euler 链)
- d_φ 的梯度穿过整个 Euler 链 → 学会"停在表面"

### 推理
- 单次 patch_based_denoise, 无外层重复
- d_φ 自动缩放步长, 不手动估计噪声水平

## Stage 1 代码改动

- `src/model/vm.py`: `get_random_indices` 改为 m>=n 时返回全量索引
- `configs/model/vm_stage1.yaml`: __target__: VelocityModule, num_train_points: 1000
- `configs/transform/vm_stage1.yaml`: noise_type: "gaussian", sigma_H: 0.02
- `configs/task/train_vm_stage1.yaml`: 15 epochs, cosine warm restart, lr=1e-4
