import jittor as jt
from .spec import ModelSpec
from .vm import VelocityModule
from .distance_module import DistanceModule
from ..data.asset import Asset
from typing import Dict, List

class CoupledVelocityModule(ModelSpec):
    """Coupled Velocity Module combining two VMs and an optional DistanceModule.
    Implements training/predict semantics compatible with the paper (K=2).
    """
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        # keep configs and instantiate two VMs
        self.K = 2
        self.vm1 = VelocityModule(model_config, transform_config)
        self.vm2 = VelocityModule(model_config, transform_config)
        # optional distance module
        self.distance_module = DistanceModule(input_dim=self.vm1.encoder.embedding_dim)
        # reuse encoder/decoder shortcuts for compatibility
        self.encoder = self.vm1.encoder
        self.decoder = self.vm1.decoder

    def set_predict(self, is_predict: bool):
        super().set_predict(is_predict)
        self.vm1.set_predict(is_predict)
        self.vm2.set_predict(is_predict)

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        # Fallback to single-VM supervised loss averaged
        loss1 = self.vm1.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        loss2 = self.vm2.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        return 0.5 * (loss1 + loss2)

    def deterministic_euler_step(self, pcl_noisy, N=3):
        # Coupled filtering per paper: for n in N: apply vm1 then vm2 with distance scaling
        B, P, d = pcl_noisy.shape
        pcl = pcl_noisy
        T = N * self.K
        for _ in range(N):
            # vm1
            feat = self.vm1.encoder(pcl)
            v0 = self.vm1.decoder(c=feat.reshape(-1, feat.shape[2])).reshape(B, P, d)
            pcl = pcl + (1.0 / T) * v0
            # vm2
            feat2 = self.vm2.encoder(pcl)
            v1 = self.vm2.decoder(c=feat2.reshape(-1, feat2.shape[2])).reshape(B, P, d)
            # distance scaling
            d_scalar = self.distance_module(feat2)
            # ensure shape
            if isinstance(d_scalar, jt.Var):
                # d_scalar: (B, P, 1)
                step = d_scalar / T
                pcl = pcl + step * v1
            else:
                pcl = pcl + (1.0 / T) * v1
        return pcl

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        
        B, Np, _ = pc_noisy.shape
        
        # VM1: 学习从 pc_noisy → pc_clean 的速度场
        loss_vm1 = self.vm1.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        
        # VM2: 学习从 X_t1 → pc_clean 的修正速度场
        # 其中 X_t1 是 VM1 一步更新后的中间状态
        # 根据 coupled filtering 理论: VM2 应该学习修正 VM1 的预测误差
        with jt.no_grad():
            v0 = self.vm1.decoder(
                c=self.vm1.encoder(pc_noisy).reshape(-1, self.vm1.encoder.embedding_dim)
            ).reshape(B, Np, 3)
            X_t1 = pc_noisy + (1.0 / self.K) * v0
        
        # VM2 基于中间状态 X_t1 学习，而非原始 pc_noisy
        loss_vm2 = self.vm2.get_supervised_loss(X_t1, X_t1, pc_clean)
        
        loss = loss_vm1 + loss_vm2
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        # Similar to VelocityModule.predict_step but using coupled filtering and repeats
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            # estimate noise level based on neighbor distances (simple heuristic)
            # compute average nearest neighbor distance
            from jittor import nn
            P = pc_next.shape[0]
            # use a small k for speed
            k = min(8, P-1)
            # compute pairwise dist to k neighbors using self.vm1.encoder.get_edge_index? fallback to knn
            # here simple heuristic: sample subset
            sample_idx = jt.arange(0, P, max(1, P//100))
            subset = pc_next[sample_idx]
            # compute kNN distances
            dists = ((subset.unsqueeze(1) - pc_next.unsqueeze(0)) ** 2).sum(-1)
            dists_k, _ = jt.topk(dists, k=k+1, dim=-1, largest=False)
            # ignore self (first col)
            avg_dist = dists_k[:, 1:].mean()
            num_repeat = 1
            if avg_dist.item() > 0.03:
                num_repeat = 3
            elif avg_dist.item() > 0.02:
                num_repeat = 2
            for _ in range(num_repeat):
                pc_next = self.deterministic_euler_step(pc_next.unsqueeze(0), N=3).squeeze(0)
            
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
        # Reuse the same data processing logic as the base VelocityModule
        # since coupled model inputs are identical.
        return self.vm1.process_fn(batch)
