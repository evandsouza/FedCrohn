import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from sources.EnvAwareGAT import EnvAwareGATCrohnModel


def test_environment_changes_prediction_and_importance_is_exposed():
    torch.manual_seed(0)
    model = EnvAwareGATCrohnModel(
        genesize=4,
        num_genes=3,
        env_dim=3,
        hidden_dim=8,
        adj_matrix=torch.eye(3, dtype=torch.float32),
    )

    x = torch.randn(2, 3, 4)
    env_low = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=torch.float32)
    env_high = torch.tensor([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], dtype=torch.float32)

    pred_low = model(x, env_low).detach()
    pred_high = model(x, env_high).detach()

    assert torch.any(torch.abs(pred_high - pred_low) > 1e-4)
    importance = model.get_env_importance()
    assert importance.shape == (3,)
    assert torch.isfinite(importance).all()


def test_combined_gene_and_env_importance_is_exposed():
    torch.manual_seed(0)
    gene_list = ["GENE_A", "GENE_B", "GENE_C"]
    model = EnvAwareGATCrohnModel(
        genesize=4,
        num_genes=3,
        env_dim=3,
        hidden_dim=8,
        adj_matrix=torch.eye(3, dtype=torch.float32),
    )
    model.geneList = gene_list

    x = torch.randn(2, 3, 4)
    model(x, torch.randn(2, 3))

    summary = model.get_combined_importance()
    assert set(summary.keys()) == {"gene_importance", "env_importance"}
    assert summary["gene_importance"].shape == (3,)
    assert summary["env_importance"].shape == (3,)
    assert torch.isfinite(summary["gene_importance"]).all()
    assert torch.isfinite(summary["env_importance"]).all()
