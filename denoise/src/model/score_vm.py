"""Score-based denoising module — ScoreDenoise (Luo & Hu, ICCV 2021).

Reuses the EdgeConv encoder + MLP decoder from the existing VelocityModule.
Key differences from StraightPCF:
  - Training target: score = NN(X_t, X_clean) - X_t  (not constant velocity X_1-X_0)
  - Inference: Langevin dynamics with annealed step size + noise injection
  - Distribution-agnostic: score points toward high-density (clean) region regardless of noise type
"""

from typing import Dict, List

import jittor as jt

from .spec import ModelSpec
from .feature import FeatureExtraction, Decoder
from .vm import get_random_indices, patch_based_denoise
from ..data.asset import Asset


def _compute_score_target(pc_state, pc_clean, chunk_knn=128):
    """Compute score s(x) = NN(x, X_clean) - x by chunked KNN (k=1).

    Avoids O(N^2) broadcast: uses streaming top-1 over clean-point chunks.
    pc_state: (B, N, 3)   pc_clean: (B, N, 3)
    Returns: (B, N, 3) detached Jittor tensor
    """
    B, N, _ = pc_state.shape
    for start in range(0, N, chunk_knn):
        end = min(start + chunk_knn, N)
        y_chunk = pc_clean[:, start:end, :]       # (B, C, 3)
        # (B, N, 1, 3) - (B, 1, C, 3) = (B, N, C, 3)
        diffs = pc_state.unsqueeze(2) - y_chunk.unsqueeze(1)
        dists = (diffs ** 2).sum(-1)               # (B, N, C)
        if start == 0:
            best_dist, best_idx = jt.topk(dists, k=1, dim=-1, largest=False)
            best_idx = best_idx.squeeze(-1) + start
        else:
            cat_dist = jt.concat([best_dist, dists], dim=-1)  # (B, N, old_topk + C)
            # Build combined indices for current best + new chunk
            idx_chunk = jt.arange(start, end).unsqueeze(0).unsqueeze(0).broadcast((B, N, end - start))
            cat_idx = jt.concat([best_idx.unsqueeze(-1), idx_chunk], dim=-1)  # (B, N, old+C)
            best_dist, top_k = jt.topk(cat_dist, k=1, dim=-1, largest=False)
            best_idx = cat_idx.gather(dim=-1, index=top_k).squeeze(-1)

    # Gather nearest clean point per state point
    batch_base = jt.arange(B).unsqueeze(1) * N   # (B, 1)
    flat_idx = (best_idx + batch_base).reshape(-1)
    nn_clean = pc_clean.reshape(-1, 3)[flat_idx].reshape(B, N, 3)

    return jt.detach(nn_clean - pc_state)


class ScoreVelocityModule(ModelSpec):
    """Score-based velocity module for point cloud denoising.

    Encoder: EdgeConv (same as VelocityModule)
    Decoder: MLP mapping features -> 3D score vectors

    The score function s(x) = ∇log p(x) points toward high-density regions (clean surface).
    Training uses denoising score matching, inference uses Langevin dynamics.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        self.dsm_sigma = cfg['dsm_sigma']

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

    # ---- Training ----

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_state'].shape[-2]
        pc_state = batch['pc_state'].reshape(-1, patch_size, 3)  # X_t
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)  # X_1

        B, N_state, d = pc_state.shape

        # Compute score target: s(x) = NN(x, X_clean) - x
        score_target = _compute_score_target(pc_state, pc_clean)

        pnt_idx = get_random_indices(N_state, self.num_train_points)

        # Forward: encode and decode
        feat = self.encoder(pc_state)  # (B, N, F)
        F_dim = feat.shape[2]

        feat = feat[:, pnt_idx, :]
        score_target = score_target[:, pnt_idx, :]

        pred_score = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d)

        loss = ((pred_score - score_target) ** 2 / self.dsm_sigma).sum(dim=-1).mean()
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    # ---- Inference (Langevin dynamics) ----

    def score_langevin_step(self, pcl_noisy, num_steps=30, alpha_init=0.15):
        """Langevin dynamics: x_{t+1} = x_t + α_t·score(x_t) + √(2α_t)·ε

        The score function drives each point toward high-density (clean surface) regions.
        Noise injection prevents getting stuck in local minima.
        Step size is annealed linearly.

        Args:
            pcl_noisy: (B, N, 3) batch of point cloud patches
            num_steps: number of Langevin iterations
            alpha_init: initial step size
        Returns:
            denoised: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl = pcl_noisy.clone()
            for step in range(num_steps):
                # Annealed step size
                progress = step / max(num_steps - 1, 1)
                alpha_t = alpha_init * (1.0 - progress)
                # Predict score
                feat = self.encoder(pcl)
                F = feat.shape[2]
                score = self.decoder(
                    c=feat.reshape(-1, F)
                ).reshape(B, N, d)
                # Langevin update
                noise = jt.randn(pcl.shape) * jt.sqrt(2.0 * alpha_t)
                pcl = pcl + alpha_t * score + noise
        return pcl

    # Compat alias: patch_based_denoise calls deterministic_euler_step by name
    def deterministic_euler_step(self, pcl_noisy, num_steps=3):
        """Backward-compatible entry for patch_based_denoise.
        Forwards to Langevin step with more iterations and appropriate alpha.
        """
        return self.score_langevin_step(pcl_noisy, num_steps=30, alpha_init=0.15)

    # ---- Predict ----

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            # Single pass through patch_based_denoise — no noise-level estimation needed.
            # The Langevin dynamics with annealed step sizes self-adapts.
            pc_next = patch_based_denoise(
                model=self,
                pcl_noisy=pc_noisy,
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=1,
            )

            # Denormalize
            asset = batch['asset'][i]
            if (asset.meta is not None
                    and 'normalize_center' in asset.meta
                    and 'normalize_scale' in asset.meta):
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
        """Same data format as VelocityModule — reuse its process_fn."""
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_state": b.meta['pc_state'],
                    "pc_noise0": b.meta['pc_noise0'],
                    "pc_clean": b.meta['pc_clean'],
                    "t_value": b.meta['t_value'],
                })
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
