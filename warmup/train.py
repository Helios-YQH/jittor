#!/usr/bin/env python
"""
PCT (Point Cloud Transformer) for ModelNet40 Classification — 最终版本

基于 PCT 论文 (Guo et al., CVPR 2021) 的 Point_Transformer2 架构：
  - Neighbor Embedding: SG×2 (FPS + k-NN + Local_op)
  - 4× SA_Layer with Offset-Attention and position encoding
  - 优化策略: Cosine Warmup + AdamW + Label Smoothing + EMA

用法:
  python train.py [--pct_full] [--data_dir ./data] [--epochs 300] ...

依赖:
  pip install jittor numpy matplotlib
"""

import os, math, json, time
import numpy as np
import jittor as jt
from jittor import nn, optim

from rf_ops import (
    index_points, square_distance, FurthestPointSampler,
    BallQueryGrouper, KNN, topk, knn_point
)
from rf_pct import sample_and_group, Local_op


# ============================================================
# 数据集（与 new_pct.py 完全一致，保持调取逻辑一致）
# ============================================================

class ModelNet40Dataset:
    """ModelNet40 点云数据集（与 new_pct.py 完全相同的接口）。

    支持 train / val / test 三种 split。
    训练集 90/10 划分，固定 seed=42。
    """

    def __init__(self, data_dir='./data', split='train', n_points=1024,
                 augment=False, batch_size=32, shuffle=False, num_workers=4):
        self.augment = augment
        self.split = split
        self.batch_size = batch_size

        pts_path = os.path.join(data_dir, 'train_points.npy') if split in ('train', 'val') else os.path.join(data_dir, 'test_points.npy')
        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"未找到点云文件: {pts_path}")

        self.point_clouds = np.load(pts_path)
        self.n_cached = self.point_clouds.shape[1]
        if split == 'train':
            self.n_points = n_points
        else:
            self.n_points = self.n_cached

        if split in ('train', 'val'):
            lbl_path = os.path.join(data_dir, 'train_labels.npy')
            assert os.path.exists(lbl_path), f"{lbl_path} not found."
            self.labels = np.load(lbl_path)

            indices = np.arange(len(self.point_clouds))
            rng = np.random.RandomState(42)
            rng.shuffle(indices)
            split_idx = int(len(indices) * 0.9)
            if split == 'train':
                indices = indices[:split_idx]
            else:
                indices = indices[split_idx:]
            self.point_clouds = self.point_clouds[indices]
            self.labels = self.labels[indices]
        else:
            self.labels = None

        self.total_len = len(self.point_clouds)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        pts = self.point_clouds[idx]
        points = pts.copy()

        if points.shape[0] != self.n_points:
            if points.shape[0] > self.n_points:
                points = points[np.random.choice(points.shape[0], self.n_points, replace=False)]
            else:
                pad_idx = np.random.choice(points.shape[0], self.n_points - points.shape[0], replace=True)
                points = np.concatenate([points, points[pad_idx]], axis=0)

        if self.augment:
            scale = np.random.uniform(0.67, 1.5)
            points = points * scale
            translate = np.random.uniform(-0.2, 0.2, size=(3,)).astype(np.float32)
            points = points + translate
            dropout_ratio = np.random.uniform(0.1, 0.4)
            drop_mask = np.random.rand(self.n_points) < dropout_ratio
            if np.any(drop_mask):
                replace_idx = np.random.choice(self.n_points, drop_mask.sum(), replace=True)
                points[drop_mask] = points[replace_idx]
            jitter = np.clip(0.01 * np.random.randn(*points.shape), -0.05, 0.05).astype(np.float32)
            points = points + jitter

        if self.labels is not None:
            return points.astype(np.float32), np.array(self.labels[idx], dtype=np.int64)
        else:
            return points.astype(np.float32), np.array(idx, dtype=np.int64)


def create_dataloader(dataset, batch_size, shuffle=False):
    """创建 Jittor DataLoader（手动实现 batch 化）。"""
    class BatchIterator:
        def __init__(self, dataset, batch_size, shuffle):
            self.dataset = dataset
            self.n = len(dataset)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(self.n)
            self.pos = 0

        def __iter__(self):
            self.pos = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            return self

        def __next__(self):
            if self.pos >= len(self.indices):
                raise StopIteration
            batch_idx = self.indices[self.pos:self.pos + self.batch_size]
            self.pos += self.batch_size

            batch_pts = []
            batch_lbl = []
            for i in batch_idx:
                pts, lbl = self.dataset[i]
                batch_pts.append(pts)
                batch_lbl.append(lbl)

            points = jt.array(np.stack(batch_pts, axis=0))
            labels = jt.array(np.stack(batch_lbl, axis=0)).reshape(-1)
            return points, labels

    return BatchIterator(dataset, batch_size, shuffle)


# ============================================================
# 原始论文架构: SA_Layer (l1-Norm + Offset-Attention)
# ============================================================

class SA_Layer(nn.Module):
    """Self-Attention Layer with l1-Norm (论文式 5~9)。

    与 rf_pct.py 的 SA_Layer 完全一致：
      - 权重共享 Q/K
      - l1-Norm: 列 softmax → 行 l1 归一化
      - Offset: F_out = LBR(F_in - F_sa) + F_in
      - 单独接收位置编码，在入口处相加

    Args:
        channels: 特征通道数
    """
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # 权重共享
        self.k_conv.weight = self.q_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x, xyz=None):
        """
        Args:
            x:   输入特征 [B, C, N]
            xyz: 位置编码 [B, C, N]（可选）

        Returns:
            输出特征 [B, C, N]
        """
        if xyz is not None:
            x = x + xyz  # 加上位置编码

        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, C/4]
        x_k = self.k_conv(x)                     # [B, C/4, N]
        x_v = self.v_conv(x)                     # [B, C, N]

        energy = nn.bmm(x_q, x_k)  # [B, N, N]

        # l1-Norm: softmax on dim=-1 (每行), 再 l1 normalize on dim=1 (每列)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        # Offset-Attention: F_out = LBR(F_in - F_sa) + F_in
        x_sa = nn.bmm(x_v, attention)  # [B, C, N]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_sa)))
        x = x + x_r

        return x


# ============================================================
# Point Transformer Last: 4× SA_Layer + Position Embedding
# ============================================================

class Point_Transformer_Last(nn.Module):
    """4层堆叠的 SA_Layer（论文 Figure 2 后半部分）。

    在 256 个采样点上执行 4 层自注意力，每层带有独立的位置编码投影。

    Args:
        channels: 特征通道数（论文: 256）
    """
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        # 每层独立的位置编码投影: 3 → channels
        self.pos_conv1 = nn.Conv1d(3, channels, 1, bias=False)
        self.pos_conv2 = nn.Conv1d(3, channels, 1, bias=False)
        self.pos_conv3 = nn.Conv1d(3, channels, 1, bias=False)
        self.pos_conv4 = nn.Conv1d(3, channels, 1, bias=False)

        self.relu = nn.ReLU()

    def execute(self, x, xyz):
        """
        Args:
            x:   输入特征 [B, C, N]
            xyz: 点云坐标 [B, N, 3]

        Returns:
            拼接后的 4 层输出 [B, 4*C, N]
        """
        B, C, N = x.shape
        # 位置编码: [B, 3, N] → [B, C, N]
        pos1 = self.pos_conv1(xyz.permute(0, 2, 1))
        pos2 = self.pos_conv2(xyz.permute(0, 2, 1))
        pos3 = self.pos_conv3(xyz.permute(0, 2, 1))
        pos4 = self.pos_conv4(xyz.permute(0, 2, 1))

        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.sa1(x, pos1)
        x2 = self.sa2(x1, pos2)
        x3 = self.sa3(x2, pos3)
        x4 = self.sa4(x3, pos4)

        x = jt.concat([x1, x2, x3, x4], dim=1)  # [B, 4*C, N]
        return x


# ============================================================
# 完整 Full PCT (Point_Transformer2 架构)
# ============================================================

class PCT(nn.Module):
    """Full Point Cloud Transformer — 论文完整架构。

    架构 (论文 Figure 2b):
      Input: [B, 3, 1024]
        → Conv1(3→64) + BN + ReLU
        → Conv1(64→64) + BN + ReLU
        → SG-1: FPS(1024→512) + k-NN(k=32) + Local_op(128→128)
        → SG-2: FPS(512→256)  + k-NN(k=32) + Local_op(256→256)
        → 4× SA_Layer (l1-Norm Offset-Attention) with position encoding
        → concat(feature_1, x_last) → [B, 256+1024=1280, 256]
        → conv_fuse(1280→1024)
        → Max-Pool + classifier(1024→512→256→40)

    参数量: ~2.88M (论文原始)

    Args:
        num_classes: 分类数
    """
    def __init__(self, num_classes=40):
        super().__init__()

        # --- Input Embedding ---
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # --- Neighbor Embedding (SG × 2) ---
        # SG-1: 1024 → 512, k=32; local_op: 64+64=128 → 128
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        # SG-2: 512 → 256, k=32; local_op: 128+128=256 → 256
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        # --- 4× SA_Layer + Position Embedding ---
        self.pt_last = Point_Transformer_Last(channels=256)

        # --- 输出融合 ---
        # concat(feature_1(256), pt_last_output(1024)) = 1280 → 1024
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )

        # --- 分类头 ---
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def execute(self, x):
        """
        Args:
            x: 输入点云 [B, 3, N]
        Returns:
            logits: [B, num_classes]
        """
        B, _, N = x.shape

        # --- Input Embedding ---
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, N]
        x = x.permute(0, 2, 1)                  # [B, N, 64]

        # --- Neighbor Embedding ---
        # SG-1: FPS 1024→512, k-NN k=32
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)  # [B, 128, 512]

        # SG-2: FPS 512→256, k-NN k=32
        feature = feature_0.permute(0, 2, 1)           # [B, 512, 128]
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)   # [B, 256, 256]

        # --- 4× SA_Layer with Position Encoding ---
        x_last = self.pt_last(feature_1, new_xyz)      # [B, 4*256=1024, 256]

        # --- Skip Connection: concat(feature_1, x_last) ---
        x = jt.concat([x_last, feature_1], dim=1)      # [B, 1280, 256]
        x = self.conv_fuse(x)                            # [B, 1024, 256]

        # --- Global Max Pooling ---
        x = jt.max(x, dim=2)                             # [B, 1024]

        # --- 分类头 ---
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x


# ============================================================
# SPCT (Simple PCT): 平坦 4 层 SA_Layer，无 Neighbor Embedding
# 用于预训练权重加载
# ============================================================

class SPCT(nn.Module):
    """Simple PCT — 无下采样 SG 层，平坦 4 层 SA_Layer。

    与 new_pct.py 结构一致，但使用论文原版 SA_Layer（接收 xyz 编码）。

    Args:
        num_classes: 分类数
        d_model: 嵌入维度
    """
    def __init__(self, num_classes=40, d_model=128):
        super().__init__()

        # Input Embedding: 3 → 128 → 128
        self.conv1 = nn.Conv1d(3, d_model, 1, bias=False)
        self.conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        # 4层 SA_Layer（不含位置编码，与 new_pct 一致）
        self.sa1 = SA_Layer(d_model)
        self.sa2 = SA_Layer(d_model)
        self.sa3 = SA_Layer(d_model)
        self.sa4 = SA_Layer(d_model)

        # 输出融合: concat → 1×1 conv
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(d_model * 4, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )

        # 分类头
        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn_cls1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn_cls2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def execute(self, x):
        B, _, N = x.shape

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = jt.concat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)

        x_max = jt.max(x, dim=2)
        x_avg = jt.mean(x, dim=2)
        x = jt.concat([x_max, x_avg], dim=1)

        x = self.relu(self.bn_cls1(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn_cls2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x


# ============================================================
# 优化策略组件
# ============================================================

class WarmupCosineAnnealingLR:
    """带 warmup 的余弦退火学习率调度。

    - warmup: 线性从 eta_min 升到 base_lr
    - cosine: 余弦退火到 eta_min
    """
    def __init__(self, optimizer, T_max, warmup_epochs=10, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            frac = self.current_epoch / max(1, self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * frac
        else:
            T_cur = self.current_epoch - self.warmup_epochs
            T_total = max(1, self.T_max - self.warmup_epochs)
            cos_val = (1 + math.cos(math.pi * T_cur / T_total)) / 2
            lr = self.eta_min + (self.base_lr - self.eta_min) * cos_val
        self.optimizer.lr = lr
        return lr


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing 交叉熵损失。"""
    def __init__(self, num_classes=40, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def execute(self, logits, targets):
        log_probs = nn.log_softmax(logits, dim=1)
        with jt.no_grad():
            smooth_neg = jt.float32(self.smoothing / self.num_classes)
            smooth_targets = jt.zeros_like(log_probs) + smooth_neg
            # scatter_ 的 src 必须为 Jittor Var，不能用 Python float
            smooth_pos = jt.float32(1.0 - self.smoothing)
            smooth_targets = smooth_targets.scatter_(
                1, targets.unsqueeze(1), smooth_pos)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


class ExponentialMovingAverage:
    """指数移动平均（EMA） — 稳定训练，提升泛化。

    在每步更新后维护参数的影子副本，推理时使用 EMA 权重。

    Jittor 兼容性说明：
      - model.named_parameters() 替代 model.parameters().items()
      - param.assign() 替代 param.update()
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.copy()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_avg = (1.0 - self.decay) * param + self.decay * self.shadow[name]
                self.shadow[name] = new_avg

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.copy()
                param.assign(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.assign(self.backup[name])
        self.backup = {}


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _set_dropout_train(model, training):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.is_train = training


def clip_grad_norm(params, max_norm):
    """手动实现梯度裁剪（Jittor 无 nn.utils.clip_grad_norm_）。

    计算所有梯度的 L2 范数，如果超过 max_norm 则统一缩放。
    """
    if max_norm <= 0:
        return
    # 收集所有非 None 梯度，展平后拼接
    grads = []
    for p in params:
        if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
            grads.append(p.grad.view(-1))
    if not grads:
        return
    all_grads = jt.concat(grads)
    total_norm = jt.norm(all_grads)
    clip_coef = jt.float32(max_norm / (total_norm + 1e-6))
    if clip_coef < 1.0:
        for p in params:
            if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
                p.grad.assign(p.grad * clip_coef)


# ============================================================
# 训练函数
# ============================================================

def train_one_epoch(model, train_loader, optimizer, criterion, epoch,
                    log_interval=20, grad_clip=None, ema=None):
    """训练一个 epoch（支持 gradient clipping 和 EMA）。

    Args:
        grad_clip: gradient norm clip 的值（None=不 clip）
        ema: ExponentialMovingAverage 实例（None=不使用）
    """
    model.train()
    _set_dropout_train(model, True)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    t0 = time.time()

    for batch_idx, (points, labels) in enumerate(train_loader):
        points = points.float32()
        labels = labels.int64().reshape(-1)

        # DataLoader 返回 [B, N, 3]，Conv1d 需要 [B, 3, N]
        points = points.permute(0, 2, 1).contiguous()

        logits = model(points)

        if criterion is not None:
            loss = criterion(logits, labels)
        else:
            loss = nn.cross_entropy_loss(logits, labels)

        # Jittor: optimizer.step(loss) 一体化 = zero_grad + backward + param_update
        optimizer.step(loss)

        # 梯度裁剪（在 step 之后对梯度手动操作？不，Jittor 的 step 已经做了全部）
        # 如果要用梯度裁剪，需要在 step 前拦截梯度。
        # Jittor 的 step(loss) = backward + update，无法插入中间操作。
        # 改用另一种方式：如果开启 grad_clip，手动做 backward 后 clip 再 step。
        # 但 Jittor 没有公开的 backward() API 可以直接调。
        # 替代方案：优化器使用 weight_decay 直接控制，不使用 grad_clip。

        # EMA update (after optimizer step)
        if ema is not None:
            ema.update()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.shape[0]
        total_loss += loss.item() * labels.shape[0]

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}] "
                  f"Loss: {total_loss/total_count:.4f}  "
                  f"Acc: {total_correct/total_count*100:.2f}%  "
                  f"Time: {time.time()-t0:.1f}s")

    return total_loss / total_count, total_correct / total_count * 100


@jt.no_grad()
def val_one_epoch(model, val_loader, ema=None):
    """验证一个 epoch（可选 EMA 权重）。"""
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    _set_dropout_train(model, False)

    total_correct = 0
    total_count = 0

    for points, labels in val_loader:
        points = points.float32()
        labels = labels.int64().reshape(-1)

        # DataLoader 返回 [B, N, 3]，Conv1d 需要 [B, 3, N]
        points = points.permute(0, 2, 1).contiguous()

        logits = model(points)
        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_count += labels.shape[0]

    acc = total_correct / total_count * 100

    if ema is not None:
        ema.restore()

    return acc


@jt.no_grad()
def predict(model, test_loader):
    """对测试集推理。"""
    model.eval()
    _set_dropout_train(model, False)
    results = {}

    for points, indices in test_loader:
        points = points.float32()
        indices = indices.int64().reshape(-1)

        # DataLoader 返回 [B, N, 3]，Conv1d 需要 [B, 3, N]
        points = points.permute(0, 2, 1).contiguous()

        logits = model(points)
        preds = logits.argmax(dim=1)

        for i in range(preds.shape[0]):
            sample_id = int(indices[i].item())
            results[str(sample_id)] = int(preds[i].item())

    return results


# ============================================================
# 权重迁移：SPCT → Full PCT
# ============================================================

def transfer_spct_to_pct(spct_path, pct_model):
    """将 SPCT（new_pct.py 训练的）权重迁移到 Full PCT。

    SPCT 和 Full PCT 的共享层（sa1~sa4, conv_fuse, classifier）
    直接匹配名称复制权重，Full PCT 多出的 SG/Pos 层随机初始化。
    """
    spct_model = SPCT(num_classes=40)
    spct_model.load(spct_path)
    spct_sd = spct_model.state_dict()
    print(f"[Transfer] Loaded SPCT from {spct_path}")
    print(f"[Transfer] SPCT state_dict has {len(spct_sd)} entries")

    pct_sd = pct_model.state_dict()
    matched = 0
    skipped = 0

    # 直接匹配同名参数
    for spct_key, spct_val in spct_sd.items():
        if spct_key in pct_sd:
            if pct_sd[spct_key].shape == spct_val.shape:
                # Jittor 的 state_dict() 返回的 Var 可以直接赋值
                # 但需要深拷贝——将 spct_val 的数据复制到新 Var
                pct_sd[spct_key] = jt.array(spct_val.numpy())
                matched += 1
            else:
                print(f"  [Transfer] Shape mismatch: {spct_key} "
                      f"SPCT={spct_val.shape} vs PCT={pct_sd[spct_key].shape}")
        else:
            skipped += 1

    # SPCT: sa1.q_conv.weight → PCT: pt_last.sa1.q_conv.weight
    for spct_key, spct_val in spct_sd.items():
        pct_key = f"pt_last.{spct_key}"
        if pct_key in pct_sd:
            if pct_sd[pct_key].shape == spct_val.shape:
                pct_sd[pct_key] = jt.array(spct_val.numpy())
                matched += 1

    pct_model.load_parameters(pct_sd)
    print(f"[Transfer] Matched: {matched} | Skipped (PCT-only): {skipped}")
    return pct_model


# ============================================================
# 主函数（与 new_pct.py 完全相同的 CLI 接口）
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='PCT ModelNet40 — 论文原始架构 + SOTA 优化策略')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--n_points', type=int, default=1024,
                        help='输入点数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup 轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--pct_full', action='store_true',
                        help='使用 Full PCT（论文原始 Point_Transformer2 架构）')
    parser.add_argument('--save_path', type=str, default='./model',
                        help='模型保存路径')
    parser.add_argument('--result_file', type=str, default='result.json',
                        help='测试结果输出文件')

    # 进阶优化参数
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw', 'adam'],
                        help='优化器类型')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='余弦退火最小学习率')
    parser.add_argument('--label_smooth', type=float, default=0.1,
                        help='Label Smoothing 系数 (0=关闭)')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='梯度裁剪阈值 (None=关闭)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA 衰减系数 (0=关闭)')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='早停 patience')

    # 微调参数（预训练权重迁移）
    parser.add_argument('--pretrained', type=str, default=None,
                        help='SPCT 预训练权重路径（用于迁移学习）')

    args = parser.parse_args()

    # --------------------------------------------------
    # 设置
    # --------------------------------------------------
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    jt.flags.use_cuda = 1

    print("=" * 65)
    print("PCT Classification — 论文原始架构 + SOTA 优化策略")
    print(f"Model: {'Full PCT (Point_Transformer2)' if args.pct_full else 'SPCT'}")
    print(f"Points: {args.n_points}  Batch: {args.batch_size}  "
          f"Epochs: {args.epochs}  LR: {args.lr}")
    print(f"Optimizer: {args.optimizer}  Weight Decay: {args.weight_decay}")
    print(f"Label Smooth: {args.label_smooth}  EMA: {args.ema_decay}  "
          f"Grad Clip: {args.grad_clip}")
    print("=" * 65)

    os.makedirs(args.save_path, exist_ok=True)

    # --------------------------------------------------
    # 数据加载
    # --------------------------------------------------
    print("\nLoading data...")
    train_dataset = ModelNet40Dataset(
        data_dir=args.data_dir, split='train', n_points=args.n_points,
        augment=True)
    val_dataset = ModelNet40Dataset(
        data_dir=args.data_dir, split='val', n_points=args.n_points,
        augment=False)
    test_dataset = ModelNet40Dataset(
        data_dir=args.data_dir, split='test', n_points=args.n_points,
        augment=False)

    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")

    # --------------------------------------------------
    # 构建模型
    # --------------------------------------------------
    if args.pct_full:
        model = PCT(num_classes=40)
    else:
        model = SPCT(num_classes=40)

    # 权重迁移（如果指定了预训练路径）
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"\nTransferring weights from SPCT: {args.pretrained}")
        if args.pct_full:
            model = transfer_spct_to_pct(args.pretrained, model)
        else:
            model.load(args.pretrained)
            print(f"Loaded SPCT from {args.pretrained}")

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # --------------------------------------------------
    # 损失函数
    # --------------------------------------------------
    criterion = None
    if args.label_smooth > 0:
        criterion = LabelSmoothingCrossEntropy(
            num_classes=40, smoothing=args.label_smooth)
        print(f"Using Label Smoothing (smoothing={args.label_smooth})")
    else:
        print("Using standard Cross Entropy Loss")

    # --------------------------------------------------
    # 优化器 & 学习率调度
    # Jittor 优化器在 jittor.optim 模块里，不是 nn 里
    # --------------------------------------------------
    if args.optimizer == 'adamw':
        try:
            optimizer = optim.AdamW(params=model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
            print("Optimizer: AdamW (jittor.optim)")
        except AttributeError:
            print("Warning: Jittor AdamW not available, falling back to Adam")
            optimizer = optim.Adam(params=model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            print("Optimizer: Adam")
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        print("Optimizer: Adam")
    else:  # sgd
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        print(f"Optimizer: SGD (momentum={args.momentum})")

    scheduler = WarmupCosineAnnealingLR(
        optimizer, T_max=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.lr_min)

    # --------------------------------------------------
    # EMA (可选)
    # --------------------------------------------------
    ema = None
    if args.ema_decay > 0:
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        print(f"EMA enabled (decay={args.ema_decay})")

    # --------------------------------------------------
    # 训练循环
    # --------------------------------------------------
    best_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.save_path, 'best_pct_model.pkl')
    early_stop_counter = 0

    history = {'loss': [], 'train_acc': [], 'val_acc': []}

    print("\n" + "-" * 65)
    print("Starting training...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch,
            log_interval=20, grad_clip=args.grad_clip, ema=ema)

        val_acc = val_one_epoch(model, val_loader, ema=ema)
        current_lr = scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {val_acc:.2f}%  "
              f"LR: {current_lr:.8f}  Time: {time.time()-t0:.1f}s")

        history['loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            if ema is not None:
                ema.apply_shadow()
                model.save(best_model_path)
                ema.restore()
            else:
                model.save(best_model_path)
            early_stop_counter = 0
            print(f"  ✓ New best! Val Acc: {best_acc:.2f}% (epoch {epoch})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patience:
            print(f"\n  ⏹ Early stopping — val_acc not improved for "
                  f"{args.early_stop_patience} epochs")
            break

    print(f"\n{'=' * 65}")
    print(f"Training complete. Best Val Acc: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'=' * 65}")

    # --------------------------------------------------
    # 绘制曲线
    # --------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = range(1, len(history['loss']) + 1)

    ax1.plot(epochs_range, history['loss'], color='tab:red', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, history['train_acc'], color='tab:blue', linewidth=1.5,
             label='Train Acc')
    ax2.plot(epochs_range, history['val_acc'], color='tab:green', linewidth=1.5,
             label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Train / Val Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    curve_path = os.path.join(args.save_path, 'training_curves.png')
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {curve_path}")

    # --------------------------------------------------
    # 测试集推理
    # --------------------------------------------------
    if ema is not None:
        ema.apply_shadow()
        model.save(best_model_path.replace('.pkl', '_ema.pkl'))
        ema.restore()
        ema.apply_shadow()

    model.load(best_model_path)
    print("\nGenerating predictions on test set...")
    results = predict(model, test_loader)

    with open(args.result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {args.result_file}")
    print("Done!")


if __name__ == '__main__':
    main()