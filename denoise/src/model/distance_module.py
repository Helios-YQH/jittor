import jittor as jt
from jittor import nn

class DistanceModule(nn.Module):
    """
    Estimate a scalar distance ratio for patches to scale velocity steps.
    """
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def execute(self, features):
        """Paper Eq.(13): d_φ = Sigmoid(Max(D_φ(X_t0)))

        Returns a per-patch scalar ∈ [0, 1] estimating the relative
        distance of the input state from the clean surface.

        Args:
            features: (B, N, F) — per-point features from encoder
        Returns:
            d: (B, 1, 1) — one scalar per patch, broadcastable to (B, N, 3)
        """
        B, N, F = features.shape
        h = self.encoder(features.reshape(B * N, F))  # (B*N, H)
        d = self.head(h).reshape(B, N, 1)              # (B, N, 1) — per-point raw
        d = jt.max(d, dim=1, keepdims=True)             # (B, 1, 1) — per-patch max
        d = jt.sigmoid(d)                                # Paper Eq.(13): sigmoid AFTER max
        return d

    def loss(self, pred_distance, true_distance_ratio):
        return ((pred_distance - true_distance_ratio) ** 2).mean()
