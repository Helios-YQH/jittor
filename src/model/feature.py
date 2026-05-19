from typing import Optional
from jittor import nn

import jittor as jt

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__()
        
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise Exception("Please assign valid activation to MLP!")
    
    def execute(self, x, edge_index):
        """
        x: (N, C)
        edge_index: (2, E)
        """
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # gather
        x_i = x[dst]  # (E, C)
        x_j = x[src]  # (E, C)
        
        # message
        tmp = jt.concat([x_i, x_j - x_i], dim=1)  # (E, 2C)
        msg = self.mlp(tmp)  # (E, out_channels)
        
        N = x.shape[0]
        out = jt.full((N, msg.shape[1]), 0)
        cnt = jt.full((N, msg.shape[1]), 0)
        
        # scatter mean
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        out = out / (cnt + 1)
        out_2 = self.lin(x)
        return out + out_2

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__(in_channels, out_channels, activation)
    
    def execute(self, x, edge_index):
        return super().execute(x, edge_index)

class FeatureExtraction(nn.Module):
    def __init__(self, k=32, input_dim=0, embedding_dim=512, distance_estimation=False):
        super().__init__()

        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation

        self.conv1 = DynamicEdgeConv(self.input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None
        )

    # ========= edge_index 构建 =========
    def get_edge_index(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        knn_idx = get_knn_idx(x, x, self.k + 1)  # (B, N, k+1)
        jt.sync_all()  # execute KNN and release intermediate GPU buffers
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N  # (B,)
        base = base.reshape(B, 1, 1)
        
        knn_idx = knn_idx + base  # (B, N, k)
        
        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base
        
        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        
        edge_index = jt.stack([src, dst], dim=0)  # (2, E)
        
        return edge_index
    
    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8) # type: ignore
    
    def execute(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape

        if self.distance_estimation:
            x = self.normalize_patch(x)

        # Compute edge indices only when needed and avoid repeated heavy base arithmetic
        # -------- conv1 --------
        edge_index1 = self.get_edge_index(x)
        x_flat = x.reshape(B * N, -1)
        x1 = self.conv1(x_flat, edge_index1)
        x1 = x1.reshape(B, N, -1)

        # -------- conv2 --------
        edge_index2 = self.get_edge_index(x1)
        x1_flat = x1.reshape(B * N, -1)
        x2 = self.conv2(x1_flat, edge_index2)
        x2 = x2.reshape(B, N, -1)

        # -------- conv3 --------
        edge_index3 = self.get_edge_index(x2)
        x_combined = jt.concat([x1, x2], dim=-1)
        x_combined_flat = x_combined.reshape(B * N, -1) # type: ignore
        x3 = self.conv3(x_combined_flat, edge_index3)
        x3 = x3.reshape(B, N, -1)

        return x3

class Decoder(nn.Module):
    
    def __init__(self, z_dim, dim, out_dim, hidden_size):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        c_dim = z_dim
        self.lin_1 = nn.Linear(c_dim, c_dim)
        self.bn_1_out = nn.BatchNorm1d(c_dim)
        
        self.lin_2 = nn.Linear(c_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, c, B=None, N=None):
        """
        c: (B*N, F)
        """
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        if self.out_dim == 1:
            net = net.reshape(B, N, -1)
            net = jt.max(net, dim=1, keepdims=True)
            net = self.lin_3(net)
            net = jt.sigmoid(net)
        else:
            net = self.lin_3(net)
        return net

def get_knn_idx(x, y, k, offset=0, chunk_size: int = 1024):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    B, N, d = x.shape
    M = y.shape[1]
    if K > M:
        K = M
    idx_list = []
    for b in range(B):
        x_b = x[b]
        y_b = y[b]
        x_b_sq = (x_b ** 2).sum(-1, keepdims=True)  # (N, 1), cached for all y chunks
        best_dist = None
        best_idx = None
        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            y_chunk = y_b[start_idx:end_idx]
            # Use expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b^T
            # Avoids materializing (N, C, d) intermediate tensor
            dot = jt.matmul(x_b, y_chunk.transpose(1, 0))  # (N, C)
            y_chunk_sq = (y_chunk ** 2).sum(-1)  # (C,)
            dist = x_b_sq + y_chunk_sq - 2 * dot  # (N, C)
            if best_dist is None:
                best_dist, best_idx = jt.topk(dist, k=K, dim=-1, largest=False)
                idx_chunk = jt.arange(start_idx, end_idx).int32().reshape(1, -1).broadcast((N, end_idx - start_idx))
                best_idx = idx_chunk.gather(dim=-1, index=best_idx)
            else:
                idx_chunk = jt.arange(start_idx, end_idx).int32().reshape(1, -1).broadcast((N, end_idx - start_idx))
                cat_dist = jt.concat([best_dist, dist], dim=-1)
                cat_idx = jt.concat([best_idx, idx_chunk], dim=-1)
                best_dist, top_k = jt.topk(cat_dist, k=K, dim=-1, largest=False)
                best_idx = cat_idx.gather(dim=-1, index=top_k)
        idx_list.append(best_idx)
        jt.sync_all()  # execute per-batch KNN, release GPU buffers
    idx = jt.stack(idx_list, dim=0)
    return idx[:, :, offset:]
