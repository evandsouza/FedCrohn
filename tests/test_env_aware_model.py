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

    assert torch.any(torch.abs(pred_high - pred_low) > 0.05)
    importance = model.get_env_importance()
    assert importance.shape == (3,)
    assert torch.isfinite(importance).all()
