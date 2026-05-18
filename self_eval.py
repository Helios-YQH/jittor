#!/usr/bin/env python
"""Self-evaluation: reserve a portion of training data for validation with full metrics.

Computes CD and P2S scores on held-out mesh data by running the complete predict pipeline.
Use this to monitor model quality during training and select the best checkpoint.

Usage:
    python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 50000
"""

import argparse
import os
import sys
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.spatial import cKDTree

# Import project modules
from src.data.asset import Asset
from src.data.augment import AugmentSample, AugmentNormalizePC, AugmentAddNoise
from src.data.datapath import Datapath, NpyLazyAsset
from src.model.parse import get_model
from src.data.transform import Transform
from src.model.vm import patch_based_denoise

# Reuse evaluation functions from evaluate.py
from evaluate import (
    chamfer_distance,
    point_to_surface_distance,
    metric_to_score,
    load_mesh_vf,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Self-evaluation for point cloud denoising")
    parser.add_argument("--task", type=str, required=True, help="Task config (same as training)")
    parser.add_argument("--split_ratio", type=float, default=0.1,
                        help="Fraction of training data to reserve for self-eval (default: 0.1)")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of points to sample per mesh")
    parser.add_argument("--noise_std", type=float, default=None,
                        help="Noise std for evaluation (default: mid of training range)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    return parser.parse_args()


def load_config(label, path):
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    return OmegaConf.to_container(OmegaConf.load(path))


def sample_and_noisify(mesh_path, num_samples, noise_std):
    """Sample points from mesh, normalize, add noise. Returns (noisy_pc, clean_pc, mesh_vertices, mesh_faces)."""
    import trimesh
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Sample points (same as AugmentSample)
    from src.data.utils import sample_vertex_groups
    sampled, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=min(1024, num_samples),
    )

    # Normalize (same as AugmentNormalizePC)
    center = (sampled.max(axis=0) + sampled.min(axis=0)) / 2.0
    sampled_centered = sampled - center
    scale = np.sqrt((sampled_centered ** 2).sum(axis=1)).max()
    clean_norm = sampled_centered / scale

    # Add noise
    noise = np.random.laplace(0, noise_std, size=clean_norm.shape)
    noisy_norm = clean_norm + noise

    return noisy_norm, clean_norm, center, scale, vertices, faces


def run_single_eval(args_tuple):
    """Evaluate a single mesh: sample, denoise, compute metrics."""
    mesh_path, model, num_samples, noise_std, patch_size, seed_k = args_tuple

    result = {"path": mesh_path, "cd_pred": None, "cd_noisy": None,
              "p2s_pred": None, "p2s_noisy": None}

    try:
        noisy_norm, clean_norm, center, scale, mv, mf = sample_and_noisify(
            mesh_path, num_samples, noise_std
        )

        # Denoise using patch_based_denoise
        pc_next = jt.array(noisy_norm.astype(np.float32))
        # Multiple repeats for thorough denoising
        for _ in range(3):
            pc_next = patch_based_denoise(
                model=model,
                pcl_noisy=pc_next,
                patch_size=patch_size,
                seed_k=seed_k,
                seed_k_alpha=1,
            )
        denoised_norm = pc_next.numpy() if isinstance(pc_next, jt.Var) else pc_next

        # Denormalize before metric computation? No — evaluate in normalized space
        # for consistency with competition evaluation
        cd_pred = chamfer_distance(denoised_norm, clean_norm, normalize=True)
        cd_noisy = chamfer_distance(noisy_norm, clean_norm, normalize=True)

        if mv is not None and mf is not None:
            p2s_pred = point_to_surface_distance(denoised_norm, mv, mf, normalize_ref_pc=clean_norm)
            p2s_noisy = point_to_surface_distance(noisy_norm, mv, mf, normalize_ref_pc=clean_norm)
        else:
            p2s_pred = None
            p2s_noisy = None

        result["cd_pred"] = cd_pred
        result["cd_noisy"] = cd_noisy
        result["p2s_pred"] = p2s_pred
        result["p2s_noisy"] = p2s_noisy

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    args = parse_args()

    # Seed
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = load_config(args.task)
    components = task['components']

    # Load model config
    model_config = load_config('model', os.path.join('configs/model', components['model']))
    transform_config = load_config('transform', os.path.join('configs/transform', components['transform']))

    # Build model
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.set_predict(True)
    model.eval()

    # Load checkpoint
    load_ckpt = task.get('load_ckpt', None)
    if load_ckpt is None:
        # Auto-find latest checkpoint in newest run subdirectory
        ckpt_root = os.path.join('experiments', components['model'])
        run_dirs = sorted([d for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))])
        if run_dirs:
            latest_run = run_dirs[-1]
            ckpt_dir = os.path.join(ckpt_root, latest_run)
            ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pkl')])
            if ckpt_files:
                load_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
                print(f"Auto-selected checkpoint: {load_ckpt}")
        if load_ckpt is None:
            print("ERROR: No checkpoint found and none specified.")
            sys.exit(1)
    model.load(load_ckpt)
    print(f"Loaded checkpoint: {load_ckpt}")

    # Load data config to get validation mesh paths
    data_config = load_config('data', os.path.join('configs/data', components['data']))
    validate_cfg = data_config.get('validate_dataset', None)
    if validate_cfg is None:
        print("ERROR: No validate_dataset in data config.")
        sys.exit(1)

    datapath = Datapath.parse(**validate_cfg['datapath'])
    all_mesh_paths = [os.path.join(datapath.input_dataset_dir, fp, datapath.data_name)
                      for fp in datapath.filepaths]

    # Reserve a subset for self-evaluation
    split_n = max(1, int(len(all_mesh_paths) * args.split_ratio))
    eval_paths = sorted(all_mesh_paths)[:split_n]
    print(f"Self-eval on {len(eval_paths)} meshes (split_ratio={args.split_ratio})")

    # Set noise level
    noise_std = args.noise_std if args.noise_std is not None else 0.0125

    # Build task list
    tasks = [(p, model, args.num_samples, noise_std, 1000, 6) for p in eval_paths]

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 8)
    print(f"Running evaluation with {n_workers} workers, noise_std={noise_std}, "
          f"num_samples={args.num_samples}...")

    # Run evaluation (single-process to avoid Jittor GPU sharing issues)
    results = []
    for t in tqdm(tasks):
        results.append(run_single_eval(t))

    # Aggregate
    cd_scores = []
    p2s_scores = []
    errors = 0

    for r in results:
        if "error" in r:
            errors += 1
            continue
        if r["cd_pred"] is not None and r["cd_noisy"] is not None:
            cd_scores.append(metric_to_score(r["cd_pred"], r["cd_noisy"]))
        if r["p2s_pred"] is not None and r["p2s_noisy"] is not None:
            p2s_scores.append(metric_to_score(r["p2s_pred"], r["p2s_noisy"]))

    # Report
    print("\n" + "=" * 60)
    print("  Self-Evaluation Results")
    print("=" * 60)
    print(f"  Samples evaluated: {len(results)}")
    print(f"  Errors:            {errors}")
    print(f"  Noise std:         {noise_std}")
    print("-" * 60)
    if cd_scores:
        print(f"  Mean CD score:     {np.mean(cd_scores):.2f} / 100")
    if p2s_scores:
        print(f"  Mean P2S score:    {np.mean(p2s_scores):.2f} / 100")
    if cd_scores and p2s_scores:
        final = 0.5 * np.mean(cd_scores) + 0.5 * np.mean(p2s_scores)
        print(f"  Final score:       {final:.2f} / 100")
    elif cd_scores:
        print(f"  Final score (CD):  {np.mean(cd_scores):.2f} / 100")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
