#!/usr/bin/env python3
"""
直观点云增强调试脚本：读取一个样本，左右对比原始点云与增强点云。
用法:
    python show.py --data_dir ./data --n_points 1024 --index 0 --seed 42
依赖:
    pip install numpy matplotlib
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 用于注册 3D 投影


# -------------------------------------------------------------------
# 增强函数（与 pct042900.py 完全相同）
# -------------------------------------------------------------------
def augment_point_cloud(points):
    """对单个点云执行一次随机增强。"""

    # 随机绕 Y 轴旋转
    theta = np.random.uniform(0, 1 / 6 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, 0, sin_t],
                  [0, 1, 0],
                  [-sin_t, 0, cos_t]], dtype=np.float32)
    points = points @ R.T

    # 随机缩放
    scale = np.random.uniform(0.8, 1.25)
    points = points * scale

    # 随机平移
    translate = np.random.uniform(-0.1, 0.1, size=(3,)).astype(np.float32)
    points = points + translate


    # 随机点丢弃（点数保持不变，用其他点替换）
    dropout_ratio = np.random.uniform(0.05, 0.1)
    drop_mask = np.random.rand(points.shape[0]) < dropout_ratio
    if np.any(drop_mask):
        replace_idx = np.random.choice(points.shape[0], drop_mask.sum(), replace=True)
        points[drop_mask] = points[replace_idx]

    # 高斯坐标抖动
    jitter = np.clip(0.01 * np.random.randn(*points.shape), -0.05, 0.05).astype(np.float32)
    points = points + jitter

    return points


# -------------------------------------------------------------------
# 数据加载与对齐
# -------------------------------------------------------------------
def load_sample(data_dir, n_points=1024, index=0):
    """
    返回:
        original: 点数对齐但未增强的点云 (n_points, 3)
        label:    类别标签 (int)
        augmented: 增强后的点云 (n_points, 3)
    """
    pts_path = os.path.join(data_dir, 'train_points.npy')
    lbl_path = os.path.join(data_dir, 'train_labels.npy')

    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"未找到点云文件: {pts_path}")
    if not os.path.exists(lbl_path):
        raise FileNotFoundError(f"未找到标签文件: {lbl_path}")

    all_pts = np.load(pts_path)          # (N, original_pts, 3)
    all_lbl = np.load(lbl_path)          # (N,)

    if index >= len(all_pts):
        raise IndexError(f"索引 {index} 超出范围（共 {len(all_pts)} 个样本）")

    raw = all_pts[index].astype(np.float32)
    label = int(all_lbl[index])

    # 点云点数对齐
    if raw.shape[0] > n_points:
        idx_choice = np.random.choice(raw.shape[0], n_points, replace=False)
        aligned = raw[idx_choice]
    else:
        idx_choice = np.random.choice(raw.shape[0], n_points, replace=True)
        aligned = raw[idx_choice]

    original = aligned.copy()
    augmented = augment_point_cloud(aligned)   # 应用全部在线增强

    return original, label, augmented

def load_test(data_dir, n_points=1024, index=0):
    testpts_path = os.path.join(data_dir, 'test_points.npy')
    test_pts = np.load(testpts_path)
    return test_pts

# -------------------------------------------------------------------
# 三维可视化
# -------------------------------------------------------------------
def visualize(original, label, augmented, index):
    """并排显示原始点云与增强点云。"""
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f"Sample {index}  |  Label = {label}  |  Points = {original.shape[0]}",
                 fontsize=14)

    # 左：原始（对齐后）
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2],
                c='steelblue', s=4, alpha=0.8)
    ax1.set_title("Original (aligned)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # 保持坐标轴等比例近似
    max_range = np.max(np.ptp(original, axis=0)) * 0.6
    mid = np.mean(original, axis=0)
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # 右：增强后
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(augmented[:, 0], augmented[:, 1], augmented[:, 2],
                c='tomato', s=4, alpha=0.8)
    ax2.set_title("Augmented")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    max_range2 = np.max(np.ptp(augmented, axis=0)) * 0.6
    mid2 = np.mean(augmented, axis=0)
    ax2.set_xlim(mid2[0] - max_range2, mid2[0] + max_range2)
    ax2.set_ylim(mid2[1] - max_range2, mid2[1] + max_range2)
    ax2.set_zlim(mid2[2] - max_range2, mid2[2] + max_range2)

    plt.tight_layout()
    plt.show()

def vis_test(points):
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle(f"Sample {points.shape[0]}")
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c='steelblue', s=4, alpha=0.8)
    ax1.set_title("points (aligned)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # 保持坐标轴等比例近似
    max_range = np.max(np.ptp(points, axis=0)) * 0.6
    mid = np.mean(points, axis=0)
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

def main():
    parser = argparse.ArgumentParser(description="可视化点云增强效果")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='包含 train_points.npy 和 train_labels.npy 的目录')
    parser.add_argument('--n_points', type=int, default=1024,
                        help='目标点数')
    parser.add_argument('--index', type=int, default=3000,
                        help='样本索引')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（固定点数对齐的随机性）')
    args = parser.parse_args()

    np.random.seed(args.seed)

    original, label, augmented = load_sample(args.data_dir, args.n_points, args.index)

    print(f"样本索引: {args.index}")
    print(f"标签:     {label}")
    print(f"原始形状: {original.shape}")
    print(f"增强形状: {augmented.shape}")

    visualize(original, label, augmented, args.index)

def main_test():
    points = load_test("./data", n_points=1024)
    vis_test(points)

if __name__ == "__main__":
    main()