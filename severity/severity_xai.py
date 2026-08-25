"""
severity_xai.py — which genes drive SEVERITY, as distinct from diagnosis.

The existing FedExplainer ranks genes by GAT attention. That attention sits
UPSTREAM of the split into two heads, so it answers "which genes does the
shared representation attend to" — it cannot separate the two tasks.

This module uses gradient attribution on each head separately, then contrasts
them. A gene that moves the severity output but not the diagnosis output is
severity-specific, and that is the clinically interesting set: genes that say
"how bad is it" rather than "is it Crohn's".

Usage:
    from severity_xai import severity_gene_importance, save_gene_report

    imp = severity_gene_importance(model, X, Y, geneList, DEVICE)
    save_gene_report(imp, "/kaggle/working/results")
"""

import numpy as np
import pandas as pd
import torch


def _grad_attribution(model, X_t, head, device, batch_size=8):
    """
    |d(output) / d(input)| * |input|, averaged over samples.

    Gradient alone says how sensitive the output is; multiplying by the input
    value gives the actual contribution of that gene in that patient. Summed
    over the feature dims to get one number per gene.
    """
    model.to(device).eval()
    n_genes = X_t.shape[1]
    total   = torch.zeros(n_genes, device=device)
    count   = 0

    for s in range(0, X_t.size(0), batch_size):
        xb = X_t[s:s + batch_size].clone().requires_grad_(True)
        diag, sev = model(xb)

        if head == "diagnosis":
            target = diag.sum()
        elif head == "severity":
            # sum of cumulative logits — total movement along the severity scale
            target = sev.sum()
        else:
            raise ValueError("head must be 'diagnosis' or 'severity'")

        model.zero_grad()
        if xb.grad is not None:
            xb.grad = None
        target.backward()

        contrib = (xb.grad * xb).abs().sum(dim=2).sum(dim=0)   # [n_genes]
        total  += contrib.detach()
        count  += xb.size(0)

    return (total / max(count, 1)).cpu().numpy()


def severity_gene_importance(model, X, Y, geneList, device,
                             cd_only=True, batch_size=8):
    """
    Returns a DataFrame ranking genes by severity-specific importance.

    cd_only=True restricts attribution to CD patients with a severity label.
    That matters: on a UC or healthy patient the severity head is unsupervised,
    so its gradients are meaningless. Attributing over them would dilute the
    signal with noise from patients the head was never trained on.

    Columns:
      sev_importance   attribution on the severity head
      diag_importance  attribution on the diagnosis head
      contrast         sev - diag, both z-scored  <- the interesting column
    """
    X = np.asarray(X, dtype=np.float32)

    if cd_only:
        keep = [i for i, y in enumerate(Y) if y[2] > 0]
        if len(keep) < 5:
            raise ValueError(f"only {len(keep)} severity-labelled samples")
        X = X[keep]

    X_t = torch.FloatTensor(X).to(device)

    sev  = _grad_attribution(model, X_t, "severity",  device, batch_size)
    diag = _grad_attribution(model, X_t, "diagnosis", device, batch_size)

    def z(a):
        sd = a.std()
        return (a - a.mean()) / (sd if sd > 1e-12 else 1.0)

    df = pd.DataFrame({
        "gene": geneList,
        "sev_importance":  sev,
        "diag_importance": diag,
        "contrast":        z(sev) - z(diag),
    })
    df["sev_rank"]  = df.sev_importance.rank(ascending=False).astype(int)
    df["diag_rank"] = df.diag_importance.rank(ascending=False).astype(int)
    return df.sort_values("sev_importance", ascending=False).reset_index(drop=True)


def pooling_gene_weights(model, X, Y, geneList, device, batch_size=8):
    """
    If the model uses attention pooling, its learned per-gene pooling weights
    are a second, independent importance signal. Returns None otherwise.
    """
    if not (getattr(model, "pool_genes", False)
            and getattr(model, "pool_mode", "") == "attention"):
        return None

    X = np.asarray(X, dtype=np.float32)
    keep = [i for i, y in enumerate(Y) if y[2] > 0]
    X_t = torch.FloatTensor(X[keep]).to(device)

    model.to(device).eval()
    acc, n = np.zeros(len(geneList)), 0
    with torch.no_grad():
        for s in range(0, X_t.size(0), batch_size):
            model(X_t[s:s + batch_size])
            w = model.pool_weights.cpu().numpy()
            acc += w.sum(axis=0); n += w.shape[0]
    return pd.DataFrame({"gene": geneList,
                         "pool_weight": acc / max(n, 1)}
                        ).sort_values("pool_weight", ascending=False)


def save_gene_report(df, out_dir="/kaggle/working/results",
                     tag="severity", top_n=50, verbose=True):
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_gene_importance.csv")
    df.to_csv(path, index=False)

    if verbose:
        print(f"=== TOP {top_n} GENES BY SEVERITY IMPORTANCE ===")
        print(df.head(top_n)[["gene", "sev_importance", "diag_rank",
                              "contrast"]].round(4).to_string(index=False))

        print(f"\n=== MOST SEVERITY-SPECIFIC (high severity, low diagnosis) ===")
        spec = df.sort_values("contrast", ascending=False).head(20)
        print(spec[["gene", "sev_rank", "diag_rank",
                    "contrast"]].to_string(index=False))

        print(f"\n=== MOST DIAGNOSIS-SPECIFIC ===")
        dspec = df.sort_values("contrast").head(20)
        print(dspec[["gene", "sev_rank", "diag_rank",
                     "contrast"]].to_string(index=False))

    print(f"\nsaved -> {path}")
    return path


def stability_across_seeds(dfs, top_n=50):
    """
    How consistent is the gene ranking across seeds?

    With 118 severity labels the ranking is not guaranteed stable, and a gene
    list that changes every run is not a finding. Report the overlap alongside
    the list; genes appearing in the top N for every seed are the ones worth
    naming in the paper.
    """
    tops = [set(d.head(top_n).gene) for d in dfs]
    common = set.intersection(*tops)
    pair = [len(a & b) / top_n for i, a in enumerate(tops) for b in tops[i+1:]]
    return {
        "n_seeds": len(dfs),
        "top_n": top_n,
        "in_all_seeds": sorted(common),
        "n_in_all": len(common),
        "mean_pairwise_overlap": float(np.mean(pair)) if pair else 1.0,
    }
