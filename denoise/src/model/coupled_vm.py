import jittor as jt
from .spec import ModelSpec
from .vm import VelocityModule, patch_based_denoise
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

    @property
    def encoder(self):
        return self.vm1.encoder

    @property
    def decoder(self):
        return self.vm1.decoder

    def freeze_backbone(self):
        """冻结 VM1/VM2，仅 DistanceModule 可训练。

        加载预训练 checkpoint 后调用，防止 VM 权重漂移。
        """
        for p in self.vm1.parameters():
            p.stop_grad()
        for p in self.vm2.parameters():
            p.stop_grad()

    def unfreeze_backbone(self):
        """解冻 VM1/VM2（全参数微调时使用）。"""
        for p in self.vm1.parameters():
            p.start_grad()
        for p in self.vm2.parameters():
            p.start_grad()

    def set_predict(self, is_predict: bool):
        super().set_predict(is_predict)
        self.vm1.set_predict(is_predict)
        self.vm2.set_predict(is_predict)

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        # Fallback to single-VM supervised loss averaged
        loss1 = self.vm1.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        loss2 = self.vm2.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        return 0.5 * (loss1 + loss2)

    def deterministic_euler_step(self, pcl_noisy, num_steps=3):
        """Paper Eq.(15): X_{t+1} = X_t + d_φ(X_init)/T * v_θ^k(X_t)

        DistanceModule scales the step size based on estimated distance to clean surface.
        """
        B, P, d = pcl_noisy.shape
        pcl = pcl_noisy

        # Compute distance scalar from initial state (paper: d_φ(X̂_M/T))
        feat_init = self.vm1.encoder(pcl)  # (B, P, F)
        d_phi = self.distance_module(feat_init)  # scalar ∈ [0,1] per batch

        T = num_steps * self.K  # 6
        for _ in range(num_steps):
            # vm1
            feat = self.vm1.encoder(pcl)
            v0 = self.vm1.decoder(c=feat.reshape(-1, feat.shape[2])).reshape(B, P, d)
            pcl = pcl + (d_phi / T) * v0
            # vm2
            feat2 = self.vm2.encoder(pcl)
            v1 = self.vm2.decoder(c=feat2.reshape(-1, feat2.shape[2])).reshape(B, P, d)
            pcl = pcl + (d_phi / T) * v1
        return pcl

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)

        B, Np, _ = pc_noisy.shape

        # VM1: 学习从 pc_noisy → pc_clean 的速度场
        loss_vm1 = self.vm1.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        # 分断图：执行并清空 loss_vm1 累积的算子，防止后续 no_grad encoder
        # 再叠加后总图超过 Jittor 融合阈值（触发 6GB+ 中间张量分配）
        jt.sync_all()
        jt.gc()

        # VM2: 学习从 X_t1 → pc_clean 的修正速度场
        with jt.no_grad():
            v0 = self.vm1.decoder(
                c=self.vm1.encoder(pc_mix).reshape(-1, self.vm1.encoder.embedding_dim)
            ).reshape(B, Np, 3)
            X_t1 = pc_noisy + (1.0 / self.K) * v0
        jt.sync_all()
        jt.gc()

        loss_vm2 = self.vm2.get_supervised_loss(X_t1, X_t1, pc_clean)

        # DistanceModule loss (paper Eq.(14), first term)
        # X_0 = pc_clean + Laplace(0, sigma_H)  [high-noise variant]
        # X_t0 = (1-t)*X_0 + t*pc_clean          [intermediate state]
        # target = ||pc_clean - X_t0|| / ||pc_clean - X_0|| = 1 - t
        sigma_H = 0.02
        import numpy as np
        high_noise_np = np.random.laplace(0, sigma_H, size=pc_clean.shape).astype(np.float32)
        high_noise = jt.array(high_noise_np)
        X_0 = pc_clean + high_noise
        t = float(np.random.uniform(0, 1))
        X_t0 = (1.0 - t) * X_0 + t * pc_clean

        with jt.no_grad():
            feat_dist = self.vm1.encoder(X_t0)  # (B, Np, F)
        d_phi_pred = self.distance_module(feat_dist)
        target_dist = 1.0 - t
        loss_dist = ((d_phi_pred - target_dist) ** 2).mean()

        loss = loss_vm1 + loss_vm2 + loss_dist
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            P = pc_next.shape[0]
            k = min(8, P - 1)
            step = max(1, P // 100)
            sample_idx = jt.arange(0, P, step)
            subset = pc_next[sample_idx]
            dists = ((subset.unsqueeze(1) - pc_next.unsqueeze(0)) ** 2).sum(-1)
            d_k, _ = jt.topk(dists, k=k + 1, dim=-1, largest=False)
            avg_dist = d_k[:, 1:].mean().item()
            if avg_dist > 0.03:
                num_repeat = 4
            elif avg_dist > 0.02:
                num_repeat = 3
            else:
                num_repeat = 2
            for _ in range(num_repeat):
                result = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=1000,
                    seed_k=6,
                    seed_k_alpha=1,
                )
                if result is not None:
                    pc_next = result
            asset = batch['asset'][i]
            if asset.meta is not None and 'normalize_center' in asset.meta and 'normalize_scale' in asset.meta:
                center = asset.meta['normalize_center']
                scale = asset.meta['normalize_scale']
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
