# Optimize v1: 激活 DistanceModule

**分支**: `optimize-v1`
**基础 checkpoint**: `experiments/vm/checkpoint_20260519_084730/checkpoint_best.pkl`

## 改动摘要

### P0: DistanceModule 激活

**问题**: DistanceModule (`src/model/distance_module.py`) 已定义并在 `CoupledVelocityModule.__init__` 中实例化，但从未在训练或推理中被调用 —— 死代码。

**论文要求** (StraightPCF Eq.13-15):
```
d_φ(X_t0) = Sigmoid(Max(D_φ(X_t0)))      # 距离标量 ∈ [0,1]
X_{t+1} = X_t + (d_φ / T) * v_θ^k(X_t)   # 缩放速度场步长
```

**改动**:
1. `distance_module.py`: 修正 `execute()` 为论文的 max→sigmoid 语义
2. `coupled_vm.py` `deterministic_euler_step`: 从初始状态计算 `d_φ`，缩放所有速度步
3. `coupled_vm.py` `training_step`: 新增 DistanceModule 损失 (论文 Eq.14)

### 训练机制

**DistanceModule 损失** (Eq.14 第一项):
- 创建高噪声变体 X_0 = pc_clean + Laplace(0, σ_H) (σ_H=0.02)
- 插值 X_t0 = (1-t)X_0 + t·pc_clean, t~U(0,1)
- Target: d_φ(X_t0) ≈ 1-t (= ||X_1-X_t0||/||X_1-X_0||)
- 冻结 VM1 encoder，仅训练 DistanceModule 参数

**注意**: 旧 checkpoint 不含 distance_module 权重（随机初始化）。首轮训练需让 d_φ 收敛（通常 1-2 epoch 即可，因为目标简单）。

## 训练命令

从已有 checkpoint 继续训练：
```bash
python run.py --task configs/task/train_vm.yaml
```

train_vm.yaml 中指定：
```yaml
load_ckpt: experiments/vm/checkpoint_20260519_084730/checkpoint_best.pkl
```

## 预期效果

- DistanceModule 预测 d_φ ≈ 1-t，噪声越大 d_φ 越接近 1（大步），接近表面时 d_φ 接近 0（小步）
- 低噪声数据不再 overshoot，高噪声数据不再 undershoot
- CD 和 P2S 指标应均有提升

## 后续优化 (P1-P3)

见 `docs/optimize_plan.md` 完整规划。
