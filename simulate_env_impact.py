import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sources.EnvAwareGAT import EnvAwareGATCrohnModel


def main(save_fig=True, out_dir="results"):
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

    baseline = np.array(baseline_pred.tolist())
    high_risk = np.array(high_risk_pred.tolist())
    delta = high_risk - baseline

    print("Low-risk environment prediction:", baseline.tolist())
    print("High-risk environment prediction:", high_risk.tolist())
    print("Predicted change in risk:")
    for i in range(len(baseline)):
        print(f"  Patient {i + 1}: {delta[i]:+.4f} absolute risk shift")

    env_imp = env_importance.cpu().numpy() if isinstance(env_importance, torch.Tensor) else np.array(env_importance)
    print("\nLearned environment importance (absolute mean weight per factor):")
    env_names = ["smoking", "diet", "pollution", "antibiotic"]
    for idx, val in enumerate(env_imp.tolist()):
        name = env_names[idx] if idx < len(env_names) else f"factor_{idx+1}"
        print(f"  {name}: {val:.4f}")

    # Create plots
    if save_fig:
        os.makedirs(out_dir, exist_ok=True)
        labels = [f"P{i+1}" for i in range(len(baseline))]
        x_idx = np.arange(len(labels))
        width = 0.35

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Grouped bar: baseline vs high-risk predictions
        axs[0].bar(x_idx - width/2, baseline, width, label='Low-risk')
        axs[0].bar(x_idx + width/2, high_risk, width, label='High-risk')
        axs[0].set_xticks(x_idx)
        axs[0].set_xticklabels(labels)
        axs[0].set_ylabel('Predicted risk')
        axs[0].set_title('Predictions per patient')
        axs[0].legend()

        # Delta bar
        axs[1].bar(x_idx, delta, color='orange')
        axs[1].set_xticks(x_idx)
        axs[1].set_xticklabels(labels)
        axs[1].set_title('Delta (high - low)')
        axs[1].axhline(0, color='k', linewidth=0.6)

        # Environment importance
        axs[2].bar(range(len(env_imp)), env_imp, color='green')
        axs[2].set_xticks(range(len(env_imp)))
        axs[2].set_xticklabels(env_names[:len(env_imp)])
        axs[2].set_title('Learned env importance')

        plt.tight_layout()
        out_path = os.path.join(out_dir, 'env_demo.png')
        fig.savefig(out_path)
        print(f"Saved figure to {out_path}")
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
