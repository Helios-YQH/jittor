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
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of points to sample per mesh")
    parser.add_argument("--noise_std", type=float, default=None,
                        help="Noise std for evaluation (default: mid of training range)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    return parser.parse_args()


def load_config(label, path=None):
    if path is None:
        path = label
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    return OmegaConf.to_container(OmegaConf.load(path))


def sample_and_noisify(mesh_path, num_samples, noise_std):
    """Sample points from mesh, normalize, add noise.

    Returns:
        noisy_norm:  (N,3) noisy point cloud in normalized space
        clean_norm:  (N,3) clean point cloud in normalized space
        center:      (3,)  normalization center (for de-normalization)
        scale:       float normalization scale
        vertices:    (V,3) raw mesh vertices (world space)
        faces:       (F,3) mesh faces
    """
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

    # Normalize to unit sphere
    center = (sampled.max(axis=0) + sampled.min(axis=0)) / 2.0
    sampled_centered = sampled - center
    scale = np.sqrt((sampled_centered ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        scale = 1.0
    clean_norm = sampled_centered / scale

    # Add noise in normalized space
    noise = np.random.laplace(0, noise_std, size=clean_norm.shape)
    noisy_norm = clean_norm + noise

    return noisy_norm, clean_norm, center, scale, vertices, faces


def run_single_eval(args_tuple):
    """Evaluate a single mesh: sample, denoise, compute metrics.

    Uses model.predict_step for consistent denoising with the official pipeline.
    """
    mesh_path, model, num_samples, noise_std = args_tuple

    result = {"path": mesh_path, "cd_pred": None, "cd_noisy": None,
              "p2s_pred": None, "p2s_noisy": None}

    try:
        noisy_norm, clean_norm, center, scale, mv, mf = sample_and_noisify(
            mesh_path, num_samples, noise_std
        )

        # Denoise using predict_step (same as official evaluation pipeline)
        pc_var = jt.array(noisy_norm.astype(np.float32)).unsqueeze(0)  # (1, N, 3)
        batch = {'pc_noisy': pc_var, 'asset': [Asset()]}
        with jt.no_grad():
            pred_list = model.predict_step(batch)
        denoised_result = pred_list[0]['pc_denoised']
        if isinstance(denoised_result, jt.Var):
            denoised_norm = denoised_result.numpy()
        elif isinstance(denoised_result, np.ndarray):
            denoised_norm = denoised_result
        else:
            denoised_norm = np.array(denoised_result)
        # predict_step returns normalized space when asset.meta is None

        # De-normalize to world space for consistent metric normalization
        clean_world = clean_norm * scale + center
        noisy_world = noisy_norm * scale + center
        denoised_world = denoised_norm * scale + center

        # CD: normalize=True uses clean_world as reference (unit-sphere normalization)
        cd_pred = chamfer_distance(denoised_world, clean_world, normalize=True)
        cd_noisy = chamfer_distance(noisy_world, clean_world, normalize=True)

        # P2S: normalize_ref_pc=clean_world applies same transform to pred/noisy and mesh vertices
        if mv is not None and mf is not None:
            p2s_pred = point_to_surface_distance(denoised_world, mv, mf, normalize_ref_pc=clean_world)
            p2s_noisy = point_to_surface_distance(noisy_world, mv, mf, normalize_ref_pc=clean_world)
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

    # Build model from config first (sets up shared references correctly)
    model_config = load_config('model', os.path.join('configs/model', components['model']))
    transform_config = load_config('transform', os.path.join('configs/transform', components['transform']))
    model = get_model(model_config=model_config, transform_config=transform_config)

    # Find checkpoint to load
    load_ckpt = task.get('load_ckpt', None)
    if load_ckpt is None:
        ckpt_root = os.path.join('experiments', components['model'])
        try:
            run_dirs = sorted([d for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))])
        except FileNotFoundError:
            run_dirs = []
        if run_dirs:
            latest_run = run_dirs[-1]
            ckpt_dir = os.path.join(ckpt_root, latest_run)
            latest_ckpt = os.path.join(ckpt_dir, 'checkpoint_latest.pkl')
            if os.path.isfile(latest_ckpt):
                load_ckpt = latest_ckpt
            else:
                ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pkl')])
                if ckpt_files:
                    load_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
            if load_ckpt:
                print(f"Auto-selected checkpoint: {load_ckpt}")
        if load_ckpt is None:
            print("ERROR: No checkpoint found and none specified.")
            sys.exit(1)

    model.load(load_ckpt)
    model.set_predict(True)
    model.eval()
    print(f"Loaded checkpoint: {load_ckpt}")

    # Load training data config for validation mesh paths
    # (self_eval needs mesh files to sample from, regardless of task config components.data)
    train_data_config = load_config('data', 'configs/data/train')
    validate_cfg = train_data_config.get('validate_dataset', None)
    if validate_cfg is None:
        print("ERROR: No validate_dataset in configs/data/train.yaml")
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
    tasks = [(p, model, args.num_samples, noise_std) for p in eval_paths]

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 8)
    print(f"Running evaluation with {n_workers} workers, noise_std={noise_std}, "
          f"num_samples={args.num_samples}...")

    # Run evaluation (single-process to avoid Jittor GPU sharing issues)
    results = []
    for i, t in enumerate(tqdm(tasks)):
        r = run_single_eval(t)
        results.append(r)
        if r.get("error") and i < 5:  # print first 5 errors
            print(f"\n  [ERROR #{i}] {r['error']}")

    # Aggregate
    cd_pred_vals = []
    cd_noisy_vals = []
    p2s_pred_vals = []
    p2s_noisy_vals = []
    cd_scores = []
    p2s_scores = []
    errors = 0

    for r in results:
        if "error" in r:
            errors += 1
            continue
        if r["cd_pred"] is not None and r["cd_noisy"] is not None:
            cd_pred_vals.append(r["cd_pred"])
            cd_noisy_vals.append(r["cd_noisy"])
            cd_scores.append(metric_to_score(r["cd_pred"], r["cd_noisy"]))
        if r["p2s_pred"] is not None and r["p2s_noisy"] is not None:
            p2s_pred_vals.append(r["p2s_pred"])
            p2s_noisy_vals.append(r["p2s_noisy"])
            p2s_scores.append(metric_to_score(r["p2s_pred"], r["p2s_noisy"]))

    # Report
    print("\n" + "=" * 60)
    print("  Self-Evaluation Results")
    print("=" * 60)
    print(f"  Samples evaluated: {len(results)}")
    print(f"  Errors:            {errors}")
    print(f"  Noise std:         {noise_std}")
    print("-" * 60)
    if cd_pred_vals:
        print(f"  CD  (pred/noisy):  {np.mean(cd_pred_vals):.6f} / {np.mean(cd_noisy_vals):.6f}")
        print(f"  Mean CD score:     {np.mean(cd_scores):.2f} / 100")
    if p2s_pred_vals:
        print(f"  P2S (pred/noisy):  {np.mean(p2s_pred_vals):.6f} / {np.mean(p2s_noisy_vals):.6f}")
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
