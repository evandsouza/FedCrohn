import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    """Lightweight graph encoder using graph-aware pooling."""

    def __init__(self, in_features, hidden_dim, num_genes):
        super().__init__()
        self.num_genes = num_genes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, adj=None):
        if adj is None:
            pooled = x.mean(dim=1)
        else:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
            deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled = torch.matmul(adj, x) / deg
            pooled = pooled.mean(dim=1)
        return self.net(pooled)


class EnvAwareGATCrohnModel(nn.Module):
    """
    Environment-aware extension for Crohn risk prediction.

    The same gene profile can be scored differently if the environmental vector changes.
    This is a demonstrator for risk modulation and learned environment contribution.
    """

    def __init__(self, genesize, num_genes, env_dim, adj_matrix=None,
                 hidden_dim=32):
        super().__init__()
        self.num_genes = num_genes
        self.env_dim = env_dim
        self.graph_encoder = GraphEncoder(genesize, hidden_dim, num_genes)

        if adj_matrix is None:
            adj = torch.eye(num_genes, dtype=torch.float32)
        else:
            adj = torch.as_tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer('adj', adj)

        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, env, adj=None):
        if adj is None:
            adj = self.adj
        gene_emb = self.graph_encoder(x, adj)
        env_emb = self.env_encoder(env)
        combined = torch.cat([gene_emb, env_emb], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits).squeeze(-1)

    def get_env_importance(self):
        """Return absolute mean importance of each environment feature."""
        with torch.no_grad():
            first_layer = self.env_encoder[0].weight
            return torch.abs(first_layer).mean(dim=0)

    def simulate_environment_shift(self, x, env_low, env_high, adj=None):
        p_low = self(x, env_low, adj=adj)
        p_high = self(x, env_high, adj=adj)
        return p_low, p_high


def example_environment_vector():
    """Example environmental factors: smoking, diet, pollution, antibiotic exposure."""
    return torch.tensor([
        [0.15, 0.20, 0.10, 0.18],
        [0.85, 0.80, 0.90, 0.82],
    ], dtype=torch.float32)
