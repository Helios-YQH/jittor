from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):
    
    num_samples: int # total number of vertices on the face to be sampled
    
    num_vertex_samples: int=0 # number of vertices to be chosen
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        assert asset.vertices is not None
        assert asset.faces is not None
        sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices = pc / scale
        # 保存归一化参数供后续可能使用
        if asset.meta is None:
            asset.meta = {}
        asset.meta['normalize_center'] = center
        asset.meta['normalize_scale'] = scale


@dataclass(frozen=True)
class AugmentNormalizePCPredict(Augment):
    """
    专门用于预测阶段的归一化
    对 sampled_vertices_noisy 进行归一化并保存参数
    """
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePCPredict':
        cls.check_keys(kwargs)
        return AugmentNormalizePCPredict(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices_noisy
        assert pc is not None, "sampled_vertices_noisy is None, cannot apply AugmentNormalizePCPredict"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices_noisy = pc / scale
        # 保存归一化参数用于反归一化
        if asset.meta is None:
            asset.meta = {}
        asset.meta['normalize_center'] = center
        asset.meta['normalize_scale'] = scale

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.laplace(0, noise_std, size=pc.shape)
        asset.sampled_vertices_noisy = pc + noise

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)

@dataclass(frozen=True)
class AugmentPatch(Augment):

    patch_size: int

    num_patches: int

    train_cvm_network: bool

    sigma_H: float=0.02

    noise_laplace_weight: float=0.6

    noise_gaussian_weight: float=0.2

    noise_uniform_weight: float=0.2

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices  # clean point cloud (N, 3)

        assert pc is not None

        N = pc.shape[0]

        # Select seed points from clean cloud for patch extraction
        seed_idx = np.random.permutation(N)[:self.num_patches]
        seed_points = pc[seed_idx]

        # Extract clean patches around seed points
        tree = cKDTree(pc)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)

        pat_clean = pc[nn_idx].astype(np.float32)  # X_1: (P, M, 3)

        # Create high-noise variant X_0 with mixed noise types
        noise_H = self._sample_mixed_noise(pat_clean.shape)
        pat_noise0 = pat_clean + noise_H  # X_0: (P, M, 3)

        # Sample t ∈ [0, 1], create intermediate state X_t = (1-t)·X_0 + t·X_1
        t = np.random.rand(self.num_patches, self.patch_size, 1).astype(np.float32)
        pat_state = (1.0 - t) * pat_noise0 + t * pat_clean  # X_t: (P, M, 3)

        # Compute seed points at each state and center patches
        # All patches centered at X_t's seed point for consistent velocity computation
        seed_clean = pat_clean[:, 0:1, :]
        seed_noise0 = pat_noise0[:, 0:1, :]
        seed_state = (1.0 - t[:, 0:1, :]) * seed_noise0 + t[:, 0:1, :] * seed_clean

        pat_clean = pat_clean - seed_state
        pat_noise0 = pat_noise0 - seed_state
        pat_state = pat_state - seed_state

        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_clean'] = pat_clean      # X_1: denoising target
        asset.meta['pc_noise0'] = pat_noise0    # X_0: high-noise variant
        asset.meta['pc_state'] = pat_state      # X_t: encoder input
        asset.meta['t_value'] = t               # for DistanceModule target = 1-t

    def _sample_mixed_noise(self, shape):
        """Sample noise from a mixture of distributions for training robustness.

        Laplace (60%): matches competition noise distribution
        Gaussian (20%): original paper noise model
        Uniform (20%): additional noise type for generalization
        """
        P, M, _ = shape

        noise_type = np.random.choice(
            [0, 1, 2], size=(P, 1, 1),
            p=[self.noise_laplace_weight, self.noise_gaussian_weight, self.noise_uniform_weight]
        )

        noise_laplace = np.random.laplace(0, self.sigma_H, size=shape).astype(np.float32)
        noise_gaussian = np.random.normal(0, self.sigma_H, size=shape).astype(np.float32)

        # Uniform within sphere of radius sigma_H
        dirs = np.random.randn(*shape).astype(np.float32)
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
        radii = np.random.rand(P, M, 1).astype(np.float32) ** (1.0 / 3.0) * self.sigma_H
        noise_uniform = (dirs * radii).astype(np.float32)

        noise = np.where(noise_type == 0, noise_laplace,
                np.where(noise_type == 1, noise_gaussian, noise_uniform))

        return noise.astype(np.float32)

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "normalize_pc_predict": AugmentNormalizePCPredict,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments