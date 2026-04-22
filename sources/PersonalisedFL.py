# sources/PersonalizedFL.py
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class PersonalizedGATCrohn(nn.Module):
    """
    GAT backbone (shared, federated) + local head (private, not aggregated).
    """
    
    def __init__(self, base_gat_model):
        super().__init__()
        self.backbone = base_gat_model
        
        # Determine backbone output size
        hidden_dim = base_gat_model.gat2.out_features
        num_genes = base_gat_model.numGenes
        backbone_out = num_genes * hidden_dim
        
        # Local head — NEVER sent to server
        self.local_head = nn.Sequential(
            nn.Linear(backbone_out, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
        # Override classifier in backbone to be identity
        self.backbone.classifier = nn.Identity()
        self.name = base_gat_model.name
    
    def forward(self, x, GET_ACT=False):
        features = self.backbone(x)  # [B, N*hidden]
        pred = self.local_head(features)
        if GET_ACT:
            return features, pred
        return pred
    
    def get_shared_params(self):
        """Only backbone params are shared with server."""
        return [val.cpu().numpy() 
                for _, val in self.backbone.state_dict().items()]
    
    def set_shared_params(self, params):
        """Only update backbone from server."""
        params_dict = zip(self.backbone.state_dict().keys(), params)
        state_dict = OrderedDict({
            k: torch.from_numpy(np.copy(v)) 
            for k, v in params_dict
        })
        self.backbone.load_state_dict(state_dict, strict=True)
    
    def fine_tune_local(self, X, Y, epochs=20, lr=1e-3):
        """Fine-tune only the local head on local data."""
        import torch.optim as optim
        
        optimizer = optim.Adam(self.local_head.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        x_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(Y)
        
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self(x_t)
            # Expand for CE loss
            pred_ce = torch.cat([torch.ones_like(pred), pred], dim=1)
            loss = loss_fn(pred_ce, y_t)
            loss.backward()
            optimizer.step()