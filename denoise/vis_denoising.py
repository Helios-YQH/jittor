#!/usr/bin/env python
"""Visualization script for denoising quality analysis.

Generates comparison figures showing noisy/denoised/clean point clouds
to expose CD and P2S convergence issues.

Usage:
    python vis_denoising.py --task configs/task/train_vm.yaml --num_samples 3 --vis_points 2000
"""

import argparse
import os
import sys
import random

import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from src.data.asset import Asset
from src.data.datapath import Datapath
from src.model.parse import get_model
from src.data.transform import Transform
from src.data.utils import sample_vertex_groups

from evaluate import chamfer_distance, point_to_surface_distance, load_mesh_vf


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize denoising quality")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--vis_points", type=int, default=2000)
    parser.add_argument("--noise_std", type=float, default=0.0125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="vis_output")
    return parser.parse_args()


def load_config(label, path=None):
    if path is None:
        path = label
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    return OmegaConf.to_container(OmegaConf.load(path))


def sample_and_noisify(mesh_path, num_samples, noise_std):
    import trimesh
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    sampled, _, _, _ = sample_vertex_groups(
        vertices=vertices, faces=faces,
        num_samples=num_samples,
        num_vertex_samples=min(1024, num_samples),
    )

    center = (sampled.max(axis=0) + sampled.min(axis=0)) / 2.0
    sampled_centered = sampled - center
    scale = np.sqrt((sampled_centered ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        scale = 1.0
    clean_norm = sampled_centered / scale

    noise = np.random.laplace(0, noise_std, size=clean_norm.shape)
    noisy_norm = clean_norm + noise

    return noisy_norm, clean_norm, center, scale, vertices, faces


def denoise_inference(model, noisy_norm):
    """Denoise using model.predict_step (same as official pipeline)."""
    pc_var = jt.array(noisy_norm.astype(np.float32)).unsqueeze(0)  # (1, N, 3)
    from src.data.asset import Asset
    batch = {'pc_noisy': pc_var, 'asset': [Asset()]}
    with jt.no_grad():
        pred_list = model.predict_step(batch)
    result = pred_list[0]['pc_denoised']
    if isinstance(result, jt.Var):
        return result.numpy()
    return np.asarray(result)


def compute_point_errors(denoised, clean, k=1):
    """Per-point nearest-neighbor distance from denoised to clean."""
    tree = cKDTree(clean)
    dists, idx = tree.query(denoised, k=k)
    return dists, idx


def main():
    args = parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = load_config(args.task)
    components = task['components']

    model_config = load_config('model', os.path.join('configs/model', components['model']))
    transform_config = load_config('transform', os.path.join('configs/transform', components['transform']))
    model = get_model(model_config=model_config, transform_config=transform_config)

    load_ckpt = task.get('load_ckpt', None)
    if load_ckpt is None:
        print("ERROR: load_ckpt not set in task config")
        sys.exit(1)

    model.load(load_ckpt)
    model.set_predict(True)
    model.eval()
    print(f"Loaded checkpoint: {load_ckpt}")

    # Load training data config for validation mesh paths
    train_data_config = load_config('data', 'configs/data/train')
    validate_cfg = train_data_config.get('validate_dataset', None)
    if validate_cfg is None:
        print("ERROR: No validate_dataset in configs/data/train.yaml")
        sys.exit(1)

    datapath = Datapath.parse(**validate_cfg['datapath'])
    all_mesh_paths = [os.path.join(datapath.input_dataset_dir, fp, datapath.data_name)
                      for fp in datapath.filepaths]

    eval_paths = sorted(all_mesh_paths)[:args.num_samples]
    print(f"Visualizing {len(eval_paths)} samples...")

    os.makedirs(args.output_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for idx, mesh_path in enumerate(eval_paths):
        print(f"\n--- Sample {idx}: {mesh_path} ---")

        noisy_norm, clean_norm, center, scale, mv, mf = sample_and_noisify(
            mesh_path, args.vis_points, args.noise_std
        )
        denoised_norm = denoise_inference(model, noisy_norm)

        # De-normalize to world space
        noisy_w = noisy_norm * scale + center
        clean_w = clean_norm * scale + center
        denoised_w = denoised_norm * scale + center

        # Metrics
        cd_pred = chamfer_distance(denoised_w, clean_w, normalize=True)
        cd_noisy = chamfer_distance(noisy_w, clean_w, normalize=True)
        if mv is not None and mf is not None:
            p2s_pred = point_to_surface_distance(denoised_w, mv, mf, normalize_ref_pc=clean_w)
            p2s_noisy = point_to_surface_distance(noisy_w, mv, mf, normalize_ref_pc=clean_w)
        else:
            p2s_pred = p2s_noisy = None

        print(f"  CD  pred/noisy: {cd_pred:.6f} / {cd_noisy:.6f}")
        if p2s_pred is not None:
            print(f"  P2S pred/noisy: {p2s_pred:.6f} / {p2s_noisy:.6f}")

        # Per-point errors
        err_denoised, _ = compute_point_errors(denoised_norm, clean_norm)
        err_noisy, _ = compute_point_errors(noisy_norm, clean_norm)

        # ===== Figure: 3-row comparison =====
        fig = plt.figure(figsize=(18, 6))

        # Subsample for cleaner visualization
        n_show = min(args.vis_points, 2000)
        show_idx = np.random.choice(args.vis_points, n_show, replace=False)

        for row, (pts, title, errs) in enumerate([
            (noisy_norm[show_idx], f"Noisy (CD={cd_noisy:.6f})", err_noisy[show_idx]),
            (denoised_norm[show_idx], f"Denoised (CD={cd_pred:.6f})", err_denoised[show_idx]),
            (clean_norm[show_idx], "Clean (GT)", np.zeros(n_show)),
        ]):
            ax = fig.add_subplot(1, 3, row + 1, projection='3d')
            colors = errs if row < 2 else 'green'
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=colors, cmap='hot', s=5, alpha=0.7
            )
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            if row < 2:
                plt.colorbar(sc, ax=ax, shrink=0.6, label='NN err to clean')

        plt.suptitle(f"Sample {idx}: CD change {cd_pred/cd_noisy:.3f}x, "
                     f"P2S change {p2s_pred/p2s_noisy:.3f}x" if p2s_pred else "",
                     fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"sample_{idx}_comparison.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> {out_path}")

        # ===== Figure: displacement vectors =====
        fig = plt.figure(figsize=(12, 5))

        # Show displacement vectors (noisy -> denoised)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        displacement = denoised_norm[show_idx] - noisy_norm[show_idx]
        disp_mag = np.linalg.norm(displacement, axis=1)
        sc1 = ax1.scatter(noisy_norm[show_idx, 0], noisy_norm[show_idx, 1], noisy_norm[show_idx, 2],
                          c=disp_mag, cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(sc1, ax=ax1, shrink=0.6, label='|displacement|')
        ax1.set_title("Noisy -> Denoised displacement")

        # Show displacement vs true displacement
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        true_disp = clean_norm[show_idx] - noisy_norm[show_idx]
        pred_disp = displacement
        angle_error = np.arccos(np.clip(
            np.sum(true_disp * pred_disp, axis=1) /
            (np.linalg.norm(true_disp, axis=1) * np.linalg.norm(pred_disp, axis=1) + 1e-8),
            -1, 1
        ))
        mean_angle = np.degrees(np.mean(angle_error))
        sc2 = ax2.scatter(pred_disp[:, 0], pred_disp[:, 1], pred_disp[:, 2],
                          c=angle_error, cmap='coolwarm', s=10, alpha=0.6)
        plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Angle error (rad)')
        ax2.set_title(f"Predicted displacement (mean angle err: {mean_angle:.1f}°)")

        plt.suptitle(f"Sample {idx}: Displacement Analysis", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"sample_{idx}_displacement.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> {out_path}")

        # ===== Figure: error distribution histogram =====
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.hist(err_noisy, bins=50, alpha=0.6, label='Noisy', color='red', density=True)
        ax.hist(err_denoised, bins=50, alpha=0.6, label='Denoised', color='blue', density=True)
        ax.axvline(x=np.mean(err_noisy), color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=np.mean(err_denoised), color='blue', linestyle='--', alpha=0.7)
        ax.set_xlabel('NN distance to clean')
        ax.set_ylabel('Density')
        ax.set_title('Per-point Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        # Show the change: positive = denoised better, negative = denoised worse
        err_diff = err_noisy - err_denoised
        ax.hist(err_diff, bins=50, alpha=0.7, color='purple')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=np.mean(err_diff), color='purple', linestyle='--',
                   label=f'Mean diff: {np.mean(err_diff):.6f}')
        ax.set_xlabel('Error reduction (noisy - denoised)')
        ax.set_ylabel('Count')
        ax.set_title('Error Change per Point\n(+ = improved, - = degraded)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Sample {idx}: Error Distribution Analysis", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"sample_{idx}_error_dist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> {out_path}")

        # ===== Figure: displacement magnitude vs direction analysis =====
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        true_disp = clean_norm - noisy_norm
        pred_disp = denoised_norm - noisy_norm
        true_mag = np.linalg.norm(true_disp, axis=1)
        pred_mag = np.linalg.norm(pred_disp, axis=1)

        # Magnitude comparison
        ax = axes[0]
        ax.scatter(true_mag, pred_mag, alpha=0.3, s=10)
        max_val = max(true_mag.max(), pred_mag.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('True displacement magnitude')
        ax.set_ylabel('Predicted displacement magnitude')
        ax.set_title('Displacement Magnitude\n(true vs predicted)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Direction error histogram
        cos_sim = np.sum(true_disp * pred_disp, axis=1) / (
            np.linalg.norm(true_disp, axis=1) * np.linalg.norm(pred_disp, axis=1) + 1e-8
        )
        angle_err = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
        ax = axes[1]
        ax.hist(angle_err, bins=40, alpha=0.7, color='orange')
        ax.axvline(x=np.mean(angle_err), color='red', linestyle='--',
                   label=f'Mean: {np.mean(angle_err):.1f}°')
        ax.set_xlabel('Angle error (degrees)')
        ax.set_ylabel('Count')
        ax.set_title('Direction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Magnitude ratio histogram
        mag_ratio = pred_mag / (true_mag + 1e-8)
        ax = axes[2]
        ax.hist(mag_ratio, bins=40, alpha=0.7, color='green', range=(0, 5))
        ax.axvline(x=1.0, color='black', linestyle='-', alpha=0.5, label='Ideal (1.0)')
        ax.axvline(x=np.mean(mag_ratio), color='green', linestyle='--',
                   label=f'Mean: {np.mean(mag_ratio):.2f}')
        ax.set_xlabel('Predicted / True magnitude ratio')
        ax.set_ylabel('Count')
        ax.set_title('Magnitude Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Sample {idx}: Displacement Quality Analysis", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"sample_{idx}_disp_quality.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> {out_path}")

    # ===== Summary figure =====
    print("\nGenerating summary...")
    # (already computed per sample above, just exit cleanly)

    return 0


if __name__ == "__main__":
    sys.exit(main())
