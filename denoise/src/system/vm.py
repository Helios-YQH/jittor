from typing import List, Dict, Optional

import numpy as np
import os

from .spec import DummySystem, DummyWriter
from ..data.asset import Asset, Exporter


def _fps_downsample(pc, target_n):
    """Farthest point sampling (numpy) for uniform spatial coverage."""
    N = pc.shape[0]
    selected = np.zeros(target_n, dtype=np.int64)
    dists = np.full(N, np.inf)
    far = 0
    for i in range(target_n):
        selected[i] = far
        centroid = pc[far]
        d = ((pc - centroid) ** 2).sum(axis=1)
        dists = np.minimum(dists, d)
        far = np.argmax(dists)
    return pc[selected]


def _interpolate_upsample(pc, target_n):
    """Upsample by adding small perturbations of nearest neighbors, avoiding exact duplicates."""
    from scipy.spatial import cKDTree
    N = pc.shape[0]
    needed = target_n - N
    tree = cKDTree(pc)
    # Pick random existing points and perturb slightly
    idxs = np.random.randint(0, N, size=needed)
    # Find nearest neighbor distances for perturbation scale
    nn_dists, _ = tree.query(pc[idxs], k=2)
    # Use half the distance to nearest neighbor as perturbation scale, clamped
    perturb_scale = np.clip(nn_dists[:, 1] if nn_dists.ndim > 1 else nn_dists, 1e-8, 0.01)
    noise = np.random.randn(needed, 3).astype(np.float64)
    noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8) * perturb_scale[:, None]
    extra = pc[idxs] + noise
    return np.concatenate([pc, extra], axis=0)

class VMWriter(DummyWriter):
    
    def __init__(self, save_dir: str="results", save_name: str="denoised", output_format: str="npy"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format
    
    def write(self, batch, prediction: List[Dict], dataset_module=None):
        pc_noisy_batch = batch['pc_noisy']
        for i, asset in enumerate(batch['asset']):
            path = asset.path
            assert path is not None, "asset path is None"
            asset_dir = os.path.dirname(path)
            normalized = os.path.normpath(asset_dir)
            parts = [p for p in normalized.split(os.sep) if p != '']
            for root_name in ('dataset_test_noisy', 'test_noisy', 'dataset_test', 'dataset_train', 'dataset_clean'):
                if root_name in parts:
                    idx = parts.index(root_name)
                    parts = parts[idx+1:]
                    break
            rel_dir = os.path.join(*parts) if parts else ''
            dirname = os.path.join(self.save_dir, rel_dir)
            os.makedirs(dirname, exist_ok=True)
            denoised = prediction[i]['pc_denoised']
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            # Ensure output point count matches noisy input; use quality-preserving resampling
            noisy_pts = getattr(asset, 'sampled_vertices_noisy', None)
            if noisy_pts is not None:
                try:
                    target_n = int(noisy_pts.shape[0])
                    if denoised_np.shape[0] != target_n:
                        if denoised_np.shape[0] > target_n:
                            # FPS-based downsampling for uniform coverage
                            denoised_np = _fps_downsample(denoised_np, target_n)
                        else:
                            # Interpolate additional points by perturbing nearest neighbors
                            denoised_np = _interpolate_upsample(denoised_np, target_n)
                except Exception:
                    pass
            if self.output_format == 'npy':
                np.save(os.path.join(dirname, f"{self.save_name}.npy"), denoised_np.astype(np.float32))
            else:
                Exporter.export_obj(denoised_np, os.path.join(dirname, f"{self.save_name}.obj"))

class VMSystem(DummySystem):
    
    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
        )
    
    # override functions in dummy system if you want to implement training/validation/prediction logic