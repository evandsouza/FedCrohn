
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATLayer(nn.Module):
    """Single Graph Attention Layer."""
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Linear transform for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        # Attention coefficients: [head, 2*out_features] -> scalar
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x, adj):
        """
        x:   [batch, num_genes, in_features]
        adj: [num_genes, num_genes]  adjacency matrix
        """
        B, N, _ = x.shape
        # Project: [B, N, num_heads * out_features]
        h = self.W(x).view(B, N, self.num_heads, self.out_features)
        # h: [B, N, H, F]
        
        # Attention: e_ij = LeakyReLU(a^T [h_i || h_j])
        h_i = h.unsqueeze(2).expand(B, N, N, self.num_heads, self.out_features)
        h_j = h.unsqueeze(1).expand(B, N, N, self.num_heads, self.out_features)
        
        e = self.leakyrelu(
            (torch.cat([h_i, h_j], dim=-1) * 
             self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1)
        )  # [B, N, N, H]
        
        # Mask non-edges with -inf before softmax
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        e = e.masked_fill(mask, float('-inf'))
        
        alpha = F.softmax(e, dim=2)  # [B, N, N, H]
        alpha = self.dropout(alpha)
        
        # Aggregate: [B, N, H, F]
        out = torch.einsum('bnjh,bjhf->bnhf', alpha, h)
        # Mean pooling over heads: [B, N, F]
        out = out.mean(dim=2)
        return out, alpha  # return alpha for explainability later


class GATCrohnModel(nn.Module):
    """
    Graph Attention Network for Crohn's disease prediction.

    Optional environment features can be included by passing env_dim > 0.
    This keeps the original graph-only workflow compatible while allowing
    a low-risk vs high-risk environment perturbation at inference time.
    """

    def __init__(self, genesize, numGenes, adj_matrix, geneList,
                 name="GAT_", num_heads=4, hidden_dim=16, env_dim=0):
        super().__init__()
        self.name = name
        self.geneInputSize = genesize
        self.numGenes = numGenes
        self.geneList = geneList
        self.env_dim = env_dim

        if adj_matrix is not None:
            self.register_buffer('adj', torch.FloatTensor(adj_matrix))
        else:
            self.register_buffer('adj', torch.FloatTensor(torch.eye(numGenes)))

        # GAT layers
        self.gat1 = GATLayer(genesize, hidden_dim, num_heads=2)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads=2)

        if env_dim > 0:
            self.env_encoder = nn.Sequential(
                nn.Linear(env_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            classifier_in = numGenes * hidden_dim + 8
        else:
            self.env_encoder = None
            classifier_in = numGenes * hidden_dim

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(classifier_in, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.attention_weights = None
        self._last_alpha = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x, adj=None, env=None):
        if adj is None:
            adj = self.adj

        # Graph encoder path
        h1, alpha1 = self.gat1(x, adj)
        h2, alpha2 = self.gat2(h1, adj)
        self.attention_weights = alpha2
        self._last_alpha = alpha2.detach()

        B = x.shape[0]
        gene_embedding = h2.reshape(B, -1)

        if self.env_encoder is not None:
            if env is None:
                raise ValueError("Environment tensor is required when env_dim > 0.")
            env_embedding = self.env_encoder(env) * 3.0
            combined = torch.cat([gene_embedding, env_embedding], dim=1)
        else:
            combined = gene_embedding

        logits = self.classifier(combined)
        return logits.squeeze(-1)

    def get_gene_importance(self):
        """Extract per-gene importance from attention weights."""
        if self.attention_weights is None:
            return None
        importance = self.attention_weights.mean(dim=[0, 3])
        return importance.sum(dim=0)

    def get_env_importance(self):
        """Return absolute mean contribution of each environment feature."""
        if self.env_encoder is None:
            return None
        with torch.no_grad():
            first_layer = self.env_encoder[0].weight
            return torch.abs(first_layer).mean(dim=0)
