
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
    GAT-based model for Crohn's disease prediction.
    Replaces BaselineNN while keeping the same interface.
    """
    
    def __init__(self, genesize, numGenes, adj_matrix, geneList, 
                 name="GAT_", num_heads=4, hidden_dim=16):
        super().__init__()
        self.name = name
        self.geneInputSize = genesize
        self.numGenes = numGenes
        self.geneList = geneList
        
        # Register adjacency as buffer (not a parameter, but saved with model)
        if adj_matrix is not None:
            self.register_buffer('adj', torch.FloatTensor(adj_matrix))
        else:
            # Default: identity (no interactions — fallback to baseline behavior)
            self.register_buffer('adj', torch.FloatTensor(adj_matrix.copy()))
        
        # GAT layers
        self.gat1 = GATLayer(genesize, hidden_dim, num_heads=2)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads=2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(numGenes * hidden_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.attention_weights = None  # stored for explainability
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
    
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.num_heads, self.out_features)
        # h: [B, N, H, F]

        head_outputs = []
        for head in range(self.num_heads):
            h_head = h[:, :, head, :]          # [B, N, F]

            # attention scores for this head only — [B, N, N]
            a_head = self.a[head]              # [2F]
            h_i = h_head.unsqueeze(2).expand(B, N, N, self.out_features)
            h_j = h_head.unsqueeze(1).expand(B, N, N, self.out_features)
            e = self.leakyrelu(
                torch.cat([h_i, h_j], dim=-1).matmul(a_head)
            )  # [B, N, N]

            # mask and softmax
            mask = (adj == 0).unsqueeze(0)     # [1, N, N]
            e = e.masked_fill(mask, float('-inf'))
            alpha = F.softmax(e, dim=2)        # [B, N, N]
            alpha = self.dropout(alpha)

            # aggregate: [B, N, F]
            out_head = torch.bmm(alpha, h_head)
            head_outputs.append(out_head)

        # mean over heads: [B, N, F]
        out = torch.stack(head_outputs, dim=0).mean(dim=0)

        # store last head's alpha for explainability (lightweight)
        self._last_alpha = alpha.detach()
        return out, alpha
    
    def get_gene_importance(self):
        """Extract per-gene importance from attention weights."""
        if self.attention_weights is None:
            return None
        # Average attention received by each gene across heads and batch
        # attn shape: [B, N, N, H] — dim 2 is "attention TO gene j"
        importance = self.attention_weights.mean(dim=[0, 3])  # [N, N]
        return importance.sum(dim=0)  # [N] — total attention each gene receives
