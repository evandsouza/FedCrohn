import torch

from sources.EnvAwareGAT import EnvAwareGATCrohnModel


def main():
    torch.manual_seed(7)

    num_genes = 12
    gene_size = 5
    env_dim = 4

    x = torch.randn(3, num_genes, gene_size)
    adj = torch.eye(num_genes, dtype=torch.float32)

    model = EnvAwareGATCrohnModel(
        genesize=gene_size,
        num_genes=num_genes,
        env_dim=env_dim,
        adj_matrix=adj,
        hidden_dim=16,
    )

    env_low = torch.tensor([
        [0.10, 0.20, 0.12, 0.15],
        [0.15, 0.18, 0.10, 0.12],
        [0.08, 0.22, 0.14, 0.11],
    ], dtype=torch.float32)

    env_high = torch.tensor([
        [0.90, 0.85, 0.95, 0.88],
        [0.88, 0.92, 0.90, 0.87],
        [0.82, 0.86, 0.91, 0.80],
    ], dtype=torch.float32)

    with torch.no_grad():
        baseline_pred = model(x, env_low, adj=adj)
        high_risk_pred = model(x, env_high, adj=adj)
        env_importance = model.get_env_importance()

    print("Low-risk environment prediction:", baseline_pred.tolist())
    print("High-risk environment prediction:", high_risk_pred.tolist())
    print("Predicted change in risk:")
    for i in range(len(baseline_pred)):
        delta = float(high_risk_pred[i] - baseline_pred[i])
        print(f"  Patient {i + 1}: +{delta:.3f} absolute risk shift")

    print("\nLearned environment importance (absolute mean weight per factor):")
    for idx, val in enumerate(env_importance.tolist()):
        print(f"  Factor {idx + 1}: {val:.4f}")


if __name__ == "__main__":
    main()
