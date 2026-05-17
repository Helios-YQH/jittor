from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        
        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim']
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        
        # target
        grad_dir_t_target = pc_clean - pc_noisy
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore
        
        loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        return loss

    def deterministic_euler_step(self, pcl_noisy, num_steps: int=3):
        """
        Deterministic Euler integration step (paper's Euler, no noise).
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        # Euler steps per the paper
        N = 3
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            # estimate noise level and determine repeat count
            def estimate_repeat(pcl):
                P = pcl.shape[0]
                k = min(8, P-1)
                # sample up to 100 points for speed
                step = max(1, P // 100)
                sample_idx = jt.arange(0, P, step)
                subset = pcl[sample_idx]
                dists = ((subset.unsqueeze(1) - pcl.unsqueeze(0)) ** 2).sum(-1)
                d_k, _ = jt.topk(dists, k=k+1, dim=-1, largest=False)
                avg_dist = d_k[:, 1:].mean()
                if avg_dist.item() > 0.03:
                    return 3
                elif avg_dist.item() > 0.02:
                    return 2
                return 1

            num_repeat = estimate_repeat(pc_noisy)
            pc_next = pc_noisy
            for _ in range(num_repeat):
                # patch-based denoise keeps tensors in Jittor
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=1000,
                    seed_k=6,
                    seed_k_alpha=1,
                )
            
            # 反归一化
            asset = batch['asset'][i]
            if asset.meta is not None and 'normalize_center' in asset.meta and 'normalize_scale' in asset.meta:
                center = asset.meta['normalize_center']
                scale = asset.meta['normalize_scale']
                # 转换为 numpy 进行反归一化
                if isinstance(pc_next, jt.Var):
                    pc_next_np = pc_next.numpy()
                else:
                    pc_next_np = pc_next
                pc_next_np = pc_next_np * scale + center
                pc_next = pc_next_np
            
            res.append({"pc_denoised": pc_next})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                    "non": {
                        "asset": b,  # 保存 asset 以便后续获取归一化参数
                    }
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    优化版本：避免 GPU->CPU 同步，保持所有操作在 GPU 上
    """
    B, N, _ = pcls.shape
    device_pts = pcls
    
    dist = jt.ones((B, N)) * 1e10
    farthest = jt.zeros((B,), dtype=jt.int32)
    
    all_indices = []
    
    for i in range(num_pnts):
        all_indices.append(farthest.unsqueeze(1))
        batch_idx = jt.arange(B)
        centroid = device_pts[batch_idx, farthest]
        d = ((device_pts - centroid.unsqueeze(1)) ** 2).sum(dim=-1)
        dist = jt.minimum(dist, d)
        farthest = jt.argmax(dist, dim=-1)
    
    indices = jt.concat(all_indices, dim=1)
    batch_idx = jt.arange(B).unsqueeze(1).broadcast(indices.shape)
    sampled = device_pts[batch_idx, indices]
    
    return sampled, indices

def knn_points(x, y, k, chunk_size: int = 1024):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    优化版本：向量化计算距离矩阵
    """
    B, P, _ = x.shape
    _, N, _ = y.shape
    if k > N:
        k = N
    
    # 向量化计算距离矩阵 (B, P, N)
    # 使用 ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    x_norm = (x ** 2).sum(dim=-1, keepdims=True)  # (B, P, 1)
    y_norm = (y ** 2).sum(dim=-1, keepdims=True)  # (B, N, 1)
    
    # 分块计算以节省显存
    if N * P > 10000000:  # 如果矩阵太大，分块处理
        dist_k = []
        idx_k = []
        for b in range(B):
            x_b = x[b]  # (P, 3)
            y_b = y[b]  # (N, 3)
            best_dist = None
            best_idx = None
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                y_chunk = y_b[start:end]  # (M, 3)
                dist = ((x_b.unsqueeze(1) - y_chunk.unsqueeze(0)) ** 2).sum(-1)  # (P, M)
                if best_dist is None:
                    best_dist, best_idx = jt.topk(dist, k=k, dim=-1, largest=False)
                    idx_chunk = jt.arange(start, end).int32().reshape(1, -1).broadcast(best_dist.shape)
                    best_idx = idx_chunk.gather(dim=-1, index=best_idx)
                else:
                    idx_chunk = jt.arange(start, end).int32().reshape(1, -1).broadcast((P, end - start))
                    cat_dist = jt.concat([best_dist, dist], dim=-1)
                    cat_idx = jt.concat([best_idx, idx_chunk], dim=-1)
                    best_dist, top_k = jt.topk(cat_dist, k=k, dim=-1, largest=False)
                    best_idx = cat_idx.gather(dim=-1, index=top_k)
            dist_k.append(best_dist)
            idx_k.append(best_idx)
        dist_k = jt.stack(dist_k, dim=0)
        idx = jt.stack(idx_k, dim=0)
    else:
        # 直接计算完整距离矩阵
        xy = jt.matmul(x, y.transpose(0, 2, 1))  # (B, P, N)
        dist = x_norm + y_norm.transpose(0, 2, 1) - 2 * xy  # (B, P, N)
        
        # 获取 top-k 最小距离
        dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    
    # 收集最近邻点
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(model: VelocityModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    优化版本：向量化操作替代 Python 循环
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    
    # 优化：使用 scatter 替代循环赋值
    all_dists = jt.ones((num_patches, N)) * 1e10
    # 向量化 scatter 操作
    all_dists = all_dists.scatter(
        dim=1,
        index=point_idxs,
        src=patch_dists
    )
        
    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
    patches_denoised = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out = model.deterministic_euler_step(curr, num_steps=3)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    
    # 优化：向量化重建点云，避免双重 Python 循环
    # 方法：使用 scatter 操作按权重分配点
    
    # 创建全局索引
    global_indices = point_idxs.reshape(-1)  # (P*M,)
    patch_indices = jt.arange(num_patches).unsqueeze(1).broadcast(point_idxs.shape).reshape(-1)  # (P*M,)
    local_indices = jt.arange(patch_size).unsqueeze(0).broadcast(point_idxs.shape).reshape(-1)  # (P*M,)
    
    # 获取每个全局点对应的最佳 patch
    best_patch_for_point = best_weights_idx[global_indices]  # (P*M,)
    
    # 判断该点是否属于这个 patch
    mask = (best_patch_for_point == patch_indices)  # (P*M,)
    
    # 收集有效的赋值
    valid_global_indices = global_indices[mask]
    valid_patch_indices = patch_indices[mask]
    valid_local_indices = local_indices[mask]
    
    # 展平 patches_denoised 用于索引
    patches_flat = patches_denoised.reshape(-1, 3)  # (P*M, 3)
    patch_offset = valid_patch_indices * patch_size
    flat_indices = patch_offset + valid_local_indices
    
    # 使用 scatter 进行向量化赋值
    pcl_out = jt.zeros((N, 3))
    pcl_out = pcl_out.scatter(
        dim=0,
        index=valid_global_indices.unsqueeze(1).broadcast((valid_global_indices.shape[0], 3)),
        src=patches_flat[flat_indices]
    )
    
    # 处理未赋值的点
    assigned_count = jt.zeros((N,), dtype=jt.int32)
    assigned_count = assigned_count.scatter_add(
        dim=0,
        index=valid_global_indices,
        src=jt.ones_like(valid_global_indices, dtype=jt.int32)
    )
    unassigned_mask = (assigned_count == 0)
    
    if unassigned_mask.sum().item() > 0:
        noisy_points = pcl_noisy.squeeze(0)
        pcl_out[unassigned_mask] = noisy_points[unassigned_mask]
    
    return pcl_out
