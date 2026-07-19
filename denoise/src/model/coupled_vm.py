import jittor as jt
from .spec import ModelSpec
from .vm import VelocityModule, patch_based_denoise
from .distance_module import DistanceModule
from ..data.asset import Asset
from typing import Dict, List


class CoupledVelocityModule(ModelSpec):
    """Coupled Velocity Module combining two VMs and an optional DistanceModule.

    Implements training/predict semantics compatible with the paper (K=2).
    Training: Eq.(7) for single VM + Eq.(10) coupling loss + Eq.(14) distance loss.
    Inference: Eq.(15) Euler integration.
    """
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        self.K = 2
        self.vm1 = VelocityModule(model_config, transform_config)
        self.vm2 = VelocityModule(model_config, transform_config)
        self.distance_module = DistanceModule(input_dim=self.vm1.encoder.embedding_dim)
        self._backbone_frozen = False
        # Phase 1 starts with DM frozen — avoid no-gradient warnings and wasted buffers
        self.freeze_distance()

    @property
    def encoder(self):
        return self.vm1.encoder

    @property
    def decoder(self):
        return self.vm1.decoder

    def freeze_backbone(self):
        """Freeze VM1/VM2 encoders and decoders. DistanceModule remains trainable."""
        for p in self.vm1.parameters():
            p.stop_grad()
        for p in self.vm2.parameters():
            p.stop_grad()
        self._backbone_frozen = True

    def unfreeze_backbone(self):
        """Unfreeze VM1/VM2."""
        for p in self.vm1.parameters():
            p.start_grad()
        for p in self.vm2.parameters():
            p.start_grad()
        self._backbone_frozen = False

    def freeze_distance(self):
        """Freeze DistanceModule. VM1/VM2 remain trainable."""
        for p in self.distance_module.parameters():
            p.stop_grad()

    def unfreeze_distance(self):
        """Unfreeze DistanceModule."""
        for p in self.distance_module.parameters():
            p.start_grad()

    def set_predict(self, is_predict: bool):
        super().set_predict(is_predict)
        self.vm1.set_predict(is_predict)
        self.vm2.set_predict(is_predict)

    def get_supervised_loss(self, pc_state, pc_noise0, pc_clean):
        """Fallback: average VM1 and VM2 supervised losses."""
        loss1 = self.vm1.get_supervised_loss(pc_state, pc_noise0, pc_clean)
        loss2 = self.vm2.get_supervised_loss(pc_state, pc_noise0, pc_clean)
        return 0.5 * (loss1 + loss2)

    def deterministic_euler_step(self, pcl_noisy, num_steps=3):
        """Paper Eq.(15): X_{t+1} = X_t + d_φ(X_init)/T * v_θ^k(X_t)

        DistanceModule scales the step size based on estimated distance to clean surface.
        """
        B, P, d = pcl_noisy.shape
        pcl = pcl_noisy

        # Compute per-patch distance scalar from initial state
        feat_init = self.vm1.encoder(pcl)        # (B, P, F)
        d_phi = self.distance_module(feat_init)  # (B, 1, 1)

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
        """Phase-aware training with Eq.(7), Eq.(10), and Eq.(14).

        Phase 1 (backbone unfrozen): VM1 + VM2 + coupling. 2 encoder calls.
        Phase 2 (backbone frozen):   DistanceModule Term 1 + Term 2. feat0 reused.
        """
        patch_size = batch['pc_state'].shape[-2]
        pc_state = batch['pc_state'].reshape(-1, patch_size, 3)     # X_t
        pc_noise0 = batch['pc_noise0'].reshape(-1, patch_size, 3)   # X_0
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)     # X_1
        t_val = batch['t_value'].reshape(-1, patch_size, 1)         # t

        B, Np, _ = pc_state.shape
        K = self.K  # 2
        target_velocity = pc_clean - pc_noise0  # X_1 - X_0 (constant)
        F_dim = self.vm1.encoder.embedding_dim

        # ---- VM1 + encoder feature (used by DM in Phase 2) ----
        feat0 = self.vm1.encoder(pc_state)
        v0 = self.vm1.decoder(c=feat0.reshape(-1, F_dim)).reshape(B, Np, 3)
        loss_vm1 = ((v0 - target_velocity) ** 2).mean()

        jt.sync_all()
        jt.gc()

        # ---- Coupling straightening loss (paper Eq.10 second term) ----
        t1 = (t_val * (K - 1) + 1) / K
        X_t1_ideal = (1 - t1) * pc_noise0 + t1 * pc_clean
        X_t1_pred = pc_state + (1.0 / K) * v0

        lambda1 = 10.0
        loss_coupling = lambda1 * ((X_t1_pred - X_t1_ideal) ** 2).mean()

        jt.sync_all()
        jt.gc()

        # ---- VM2: refine velocity from predicted next state ----
        with jt.no_grad():
            X_t1_input = X_t1_pred
        feat1 = self.vm2.encoder(X_t1_input)
        v1 = self.vm2.decoder(c=feat1.reshape(-1, F_dim)).reshape(B, Np, 3)
        loss_vm2 = ((v1 - target_velocity) ** 2).mean()

        jt.sync_all()
        jt.gc()

        # ---- DistanceModule (Phase 2 only: backbone frozen) ----
        # Phase 1: backbone unfrozen → skip DM, VMs are being pretrained
        # Phase 2: backbone frozen → DM trained with feat0 detached from frozen encoder
        loss_dist_term1 = 0.0
        loss_dist_term2 = 0.0
        if self._backbone_frozen:
            # feat0 already computed by frozen VM1 — no grad tracked, safe to reuse
            d_phi_pred = self.distance_module(feat0)
            target_dist = 1.0 - t_val.mean(dim=1, keepdims=True)
            loss_dist_term1 = ((d_phi_pred - target_dist) ** 2).mean()

            # Term 2: Euler integration — d_φ gradients flow through VM chain
            lambda2 = 200.0
            X_bar = pc_state
            T = 3 * K  # 6 total steps
            for _ in range(3):
                with jt.no_grad():
                    f0 = self.vm1.encoder(X_bar)
                    v0_step = self.vm1.decoder(c=f0.reshape(-1, F_dim)).reshape(B, Np, 3)
                X_bar = X_bar + (d_phi_pred / T) * v0_step
                with jt.no_grad():
                    f1 = self.vm2.encoder(X_bar)
                    v1_step = self.vm2.decoder(c=f1.reshape(-1, F_dim)).reshape(B, Np, 3)
                X_bar = X_bar + (d_phi_pred / T) * v1_step
            loss_dist_term2 = lambda2 * ((X_bar - pc_clean) ** 2).mean()

        loss = loss_vm1 + loss_vm2 + loss_coupling + loss_dist_term1 + loss_dist_term2
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
        return self.vm1.process_fn(batch)
