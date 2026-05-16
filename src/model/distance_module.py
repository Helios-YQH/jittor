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
        # features: (B, N, F) or (B*N, F)
        if len(features.shape) == 3:
            B, N, F = features.shape
            features = features.reshape(B * N, F)
            squeeze_back = True
        else:
            squeeze_back = False
        h = self.encoder(features)
        d = self.head(h)
        d = jt.sigmoid(d)
        if squeeze_back:
            d = d.reshape(B, N, 1)
        return d

    def loss(self, pred_distance, true_distance_ratio):
        return ((pred_distance - true_distance_ratio) ** 2).mean()
