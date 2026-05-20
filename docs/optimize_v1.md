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

**未改动的文件**（与 master 一致）:
- `feature.py`: 无 `chunked_forward`，回到 master 原始执行逻辑
- `vm.py`: `self.encoder(pc_mix)` 直接调用，无分块
- `spec.py`: 无额外 `jt.sync_all()` flush

### 训练机制

**DistanceModule 损失** (Eq.14 第一项):
- 创建高噪声变体 X_0 = pc_clean + Laplace(0, σ_H) (σ_H=0.02)
- 插值 X_t0 = (1-t)X_0 + t·pc_clean, t~U(0,1)
- Target: d_φ(X_t0) ≈ 1-t (= ||X_1-X_t0||/||X_1-X_0||)

**全参数训练**: VM1/VM2 不解冻，DistanceModule 仅 81KB / 4.7% 总参数，冻结对 forward 无加速。

**注意**: 旧 checkpoint 不含 distance_module 权重（随机初始化）。

---

## NOTICE: OOM 问题与 MPI 方案

### 现象

单卡 `batch_size=72` 时，conv3 层 fused operator 分配 `float32[9216000,192]`(6.6GB) 失败：
```
Execute fused operator(29508/55968) failed.
Unable to alloc cuda device memory for size 7077888000
```

### 根因

Jittor 惰性执行在两次 encoder 调用间累积 **100K+ 算子**。图融合器打包时产生超大型中间张量，超过单卡 24GB 显存。

**与 DistanceModule 本身无关**（DistanceModule 仅 81KB 参数），纯粹是 batch_size=72 时单次前向图超出 Jittor 融合阈值。

### 修复

**代码层面**（已合入 `coupled_vm.py`）：
- `loss_vm1` 后加 `jt.sync_all() + jt.gc()` 分断图
- `no_grad` block 后加 `jt.sync_all() + jt.gc()` 分断图
- 将 100K 算子的巨图切成 30K+30K 的子图，fusion 中间张量不再超限

**配置层面**（推荐）：使用 MPI 多卡分布式训练，batch 自动分摊。

---

## 训练命令

### 6 卡 MPI 分布式（推荐）

```bash
mpirun -np 6 python run.py --task configs/task/train_vm.yaml
```

`configs/data/train.yaml` 中 `batch_size: 72` 为**全局总 batch**，Jittor 自动平分到每卡（每卡 12 个样本）。

Jittor 自动处理：
- 数据集自动拆分（每卡读不同子集）
- `model.mpi_param_broadcast(root=0)` 同步初始参数
- 梯度 all-reduce（内置在 `optimizer.step()` 中）

### 单卡调试

```bash
# 小 batch 防 OOM
python run.py --task configs/task/train_vm.yaml
# 或临时修改 train.yaml: batch_size: 12
```

### batch_size 参考

| 卡数 | batch_size（全局） | 每卡样本数 | 显存占用 |
|------|------------------|-----------|---------|
| 1 | 12 | 12 | ~8GB |
| 1 | 24 | 24 | ~12GB |
| 6 | 72 | 12 | ~8GB/卡 |
| 6 | 144 | 24 | ~12GB/卡 |

> 注意：`validate_dataset.batch_size: 6` 为每卡 6 个，6 卡时全局 36 个。

## 预期效果

- DistanceModule 预测 d_φ ≈ 1-t，噪声越大 d_φ 越接近 1（大步），接近表面时 d_φ 接近 0（小步）
- 低噪声数据不再 overshoot，高噪声数据不再 undershoot
- CD 和 P2S 指标应均有提升

## 后续优化 (P1-P3)

见 `docs/optimize_plan.md` 完整规划。
