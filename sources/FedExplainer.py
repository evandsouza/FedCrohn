# sources/FedExplainer.py
import numpy as np
from collections import defaultdict

class LocalExplainer:
    """
    Runs on each FL client after training.
    Extracts gene importance using GAT attention weights.
    """
    
    def __init__(self, model, geneList):
        self.model = model
        self.geneList = geneList
    
    def get_attention_importance(self, X):
        """
        Forward pass to collect attention weights,
        then compute per-gene importance scores.
        Returns: dict {gene_name: importance_score}
        """
        import torch
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X)
            _ = self.model(x_tensor)
        
        importance = self.model.get_gene_importance()
        if importance is None:
            return {}
        
        importance_np = importance.cpu().numpy()
        return {self.geneList[i]: float(importance_np[i]) 
                for i in range(len(self.geneList))}
    
    def serialize(self, importance_dict):
        """Convert to list aligned with geneList for transmission."""
        return np.array([importance_dict.get(g, 0.0) 
                         for g in self.geneList])


class GlobalExplainer:
    """
    Runs on the FL server.
    Aggregates local importance scores from all clients.
    """
    
    def __init__(self, geneList):
        self.geneList = geneList
        self.round_importances = []  # history per round
    
    def aggregate(self, client_importances, client_sizes):
        """
        Weighted average of importance vectors from clients.
        
        client_importances: list of np.arrays [num_genes]
        client_sizes:       list of ints (num samples per client)
        """
        total = sum(client_sizes)
        weighted = np.zeros(len(self.geneList))
        
        for imp, size in zip(client_importances, client_sizes):
            weighted += imp * (size / total)
        
        self.round_importances.append(weighted)
        return weighted
    
    def get_top_genes(self, n=20):
        """Return top-n most important genes across all rounds."""
        if not self.round_importances:
            return []
        mean_imp = np.mean(self.round_importances, axis=0)
        top_idx = np.argsort(mean_imp)[::-1][:n]
        return [(self.geneList[i], float(mean_imp[i])) for i in top_idx]
    
    def save_report(self, path="results/gene_importance.txt"):
        """Save gene ranking to file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        top = self.get_top_genes(50)
        with open(path, 'w') as f:
            f.write("Rank\tGene\tImportance\n")
            for rank, (gene, score) in enumerate(top, 1):
                f.write(f"{rank}\t{gene}\t{score:.6f}\n")
        print(f"Gene importance report saved to {path}")