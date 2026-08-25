"""
severity_model.py — severity-aware extension to the existing FedCrohn GAT.

Adds a second output branch to GATCrohnModel without touching GATLayer, the
adjacency buffer, or the federated loop. Drop this in AFTER your existing
Cell 6 / Cell 6b patches — it overrides forward() again, deliberately.

    from severity_model import (attach_severity_heads, patch_wrapper_multitask,
                                fedavg_multitask, severity_metrics)

    attach_severity_heads(GATCrohnModel)      # class-level patch, do once
    patch_wrapper_multitask(NNwrapper, DEVICE)

    net = GATCrohnModel(3, 691, adj, geneList)   # genesize=3 from the loader
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# 1. Architecture — shared trunk + diagnosis head + CORAL severity head
# ═════════════════════════════════════════════════════════════════════════════
def attach_severity_heads(model_cls, n_sev_levels=4, pool_genes=False,
                          pool_mode="mean"):
    """
    Replaces GATCrohnModel.__init__ tail and forward with a two-head version.

    CORAL rather than 4-way softmax: SES-CD levels are ORDERED, so predicting
    'severe' when the truth is 'mild' should cost more than predicting
    'moderate'. One shared weight vector + K-1 monotone biases guarantees the
    cumulative probabilities stay ordered.

    Head sizing (N=691, hidden=16):
      pool_genes=False              flatten -> Linear(11056, 64)  ~708k params
      pool_genes=True, "mean"       mean    -> Linear(16, 64)     ~1.5k params
      pool_genes=True, "attention"  learned -> Linear(16, 64)     ~1.5k params

    Why the modes matter for DP: noise is injected PER PARAMETER, so the noise
    vector norm scales with sqrt(d). At d=708k it swamps the clipped gradient
    sum for any usable sigma. Shrinking d is the only lever — but "mean"
    pooling weights every gene equally and empirically destroys utility even
    at sigma=0 (MCC 0.05 vs 0.28 flattened). "attention" keeps the same
    parameter count but learns INPUT-DEPENDENT per-gene weights, which is the
    thing mean pooling throws away.

    Any pooled variant must be benchmarked at sigma=0 before its DP numbers
    mean anything.
    """
    orig_init = model_cls.__init__

    def __init__(self, genesize, numGenes, adj_matrix, geneList,
                 name="GAT_", num_heads=4, hidden_dim=16, **kw):
        orig_init(self, genesize, numGenes, adj_matrix, geneList,
                  name=name, num_heads=num_heads, hidden_dim=hidden_dim, **kw)

        self.n_sev_levels = n_sev_levels
        self.pool_genes   = pool_genes
        self.pool_mode    = pool_mode

        if pool_genes and pool_mode == "attention":
            # one score per gene, computed from that gene's own embedding
            self.pool_a = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        trunk_in = hidden_dim if pool_genes else numGenes * hidden_dim
        self.trunk = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(trunk_in, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        self.diag_head = nn.Linear(64, 1)

        self.sev_fc   = nn.Linear(64, 1, bias=False)
        self.sev_bias = nn.Parameter(torch.zeros(n_sev_levels - 1))

        if hasattr(self, "classifier"):
            del self.classifier

        for m in [self.trunk, self.diag_head, self.sev_fc]:
            for mod in m.modules() if hasattr(m, "modules") else [m]:
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        mod.bias.data.fill_(0.01)

    def forward(self, x, GET_ACT=False):
        out, _     = self.gat1(x, self.adj)
        out        = F.leaky_relu(out)
        out, attn2 = self.gat2(out, self.adj)
        self.attention_weights = attn2.detach()

        if self.pool_genes and self.pool_mode == "attention":
            scores = out.matmul(self.pool_a)                  # [B, N]
            w      = F.softmax(scores, dim=1).unsqueeze(-1)   # [B, N, 1]
            self.pool_weights = w.detach().squeeze(-1)        # for explainability
            z = (w * out).sum(dim=1)                          # [B, hidden]
        elif self.pool_genes:
            z = out.mean(dim=1)                               # [B, hidden]
        else:
            z = out.view(out.size(0), -1)                     # [B, N*hidden]

        h    = self.trunk(z)
        diag = self.diag_head(h)
        sev  = self.sev_fc(h) + self.sev_bias

        if GET_ACT:
            return out, (diag, sev)
        return diag, sev

    model_cls.__init__ = __init__
    model_cls.forward  = forward
    return model_cls


# ═════════════════════════════════════════════════════════════════════════════
# 2. Masked multi-task loss
# ═════════════════════════════════════════════════════════════════════════════
def multitask_loss(diag_logit, sev_logits, y_diag, y_sev, sev_mask,
                   pos_weight=None, lam=0.3):
    """
    L = BCE(diagnosis) + lam * masked_CORAL(severity)

    sev_mask is 1 only for CD samples with a recorded SES-CD. Every UC and
    nonIBD sample contributes 0 to the severity term.
    """
    L_diag = F.binary_cross_entropy_with_logits(
        diag_logit.squeeze(-1), y_diag.float(),
        pos_weight=pos_weight, reduction="mean"
    )

    B, K   = sev_logits.shape
    levels = torch.arange(K, device=sev_logits.device).unsqueeze(0)   # [1, K]
    targets = (y_sev.unsqueeze(1) > levels).float()                   # [B, K]

    per_sample = F.binary_cross_entropy_with_logits(
        sev_logits, targets, reduction="none"
    ).sum(dim=1)                                                      # [B]

    # clamp guards the real case where a client batch has zero labelled
    # severity samples — without it you get NaN grads that silently poison
    # every subsequent FedAvg round.
    denom = sev_mask.sum().clamp(min=1.0)
    L_sev = (per_sample * sev_mask).sum() / denom

    return L_diag + lam * L_sev, L_diag.item(), L_sev.item()


def coral_to_level(sev_logits):
    """CORAL cumulative logits -> integer level 0..K-1."""
    return (torch.sigmoid(sev_logits) > 0.5).sum(dim=1)


def set_severity_prior(model, y_sev, sev_mask, n_levels=4):
    """
    Initialise sev_bias to the empirical cumulative log-odds of the TRAINING
    labels. Call once per fold, on the client/global model, before training.

    Why this matters: sev_fc produces ONE scalar per sample, so with
    sev_bias = zeros every cumulative logit is identical and the head can only
    ever emit level 0 or level K-1 — intermediate levels are structurally
    unreachable until the biases separate. Starting them at the empirical
    marginal removes that dead zone immediately.

    Pass only training-fold labels. Using all labels leaks the test
    distribution into the initialisation.
    """
    m = np.asarray(sev_mask).astype(bool)
    y = np.asarray(y_sev)[m]
    if len(y) < 4:
        return model

    biases = []
    for k in range(n_levels - 1):
        p = float((y > k).mean())
        p = min(max(p, 1.0 / (len(y) + 2)), 1.0 - 1.0 / (len(y) + 2))  # clip
        biases.append(np.log(p / (1.0 - p)))

    with torch.no_grad():
        model.sev_bias.copy_(torch.tensor(biases, dtype=torch.float32,
                                          device=model.sev_bias.device))
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 3. Training / prediction patch for NNwrapper
# ═════════════════════════════════════════════════════════════════════════════
def compute_epsilon(n_samples, batch_size, epochs, rounds, sigma, delta=1e-5):
    """
    (eps, delta) for the sampled Gaussian mechanism via RDP (Mironov et al.),
    integer orders only. Pure numpy — no Opacus needed, which matters because
    Kaggle sessions here run with internet disabled.

    n_samples is the SMALLEST client's local dataset: privacy is per-client,
    so the worst-case client sets the reported guarantee.

    This bound is slightly loose versus TF-Privacy / Opacus. State the
    accountant you used when reporting the number.
    """
    from math import comb, exp, log

    if sigma <= 0:
        return float("inf")

    q     = min(batch_size / max(n_samples, 1), 1.0)
    steps = int(np.ceil(n_samples / batch_size)) * epochs * rounds

    best = float("inf")
    for alpha in range(2, 128):
        try:
            s = 0.0
            for k in range(alpha + 1):
                s += (comb(alpha, k) * ((1 - q) ** (alpha - k)) * (q ** k)
                      * exp(k * (k - 1) / (2 * sigma ** 2)))
            rdp = log(s) / (alpha - 1) * steps
            eps = rdp + log(1.0 / delta) / (alpha - 1)
        except (OverflowError, ValueError):
            # high orders overflow for small sigma; the minimum sits at low
            # alpha in that regime anyway, so stopping here is safe
            break
        best = min(best, eps)
    return best


def patch_wrapper_multitask(wrapper_cls, device, dp_sigma=0.0,
                            dp_per_example=True):
    """
    Replaces NNwrapper.fit/predict with multi-task versions.

    Y must be a list of (y_diag, y_sev, sev_mask) tuples from the loader.

    dp_sigma > 0 enables DP-SGD. With dp_per_example=True each example's
    gradient is clipped SEPARATELY before summation, which is what actually
    bounds one patient's influence. Batch-level clipping (the default in the
    original pipeline) does NOT yield a valid (eps, delta) guarantee no matter
    how much noise is added — do not report an epsilon computed against it.

    Per-example clipping costs roughly batch_size times more compute.
    """

    def _dp_step(self, X_t, y_diag, y_sev, s_mask, idx, opt, lam, clip):
        """One DP-SGD step: per-example clip -> sum -> add noise -> average."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        accum  = [torch.zeros_like(p) for p in params]

        for i in idx:
            opt.zero_grad()
            diag, sev = self.model(X_t[i:i + 1])
            loss, _, _ = multitask_loss(diag, sev, y_diag[i:i + 1],
                                        y_sev[i:i + 1], s_mask[i:i + 1],
                                        None, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
            for a, p in zip(accum, params):
                if p.grad is not None:
                    a += p.grad

        opt.zero_grad()
        B = max(len(idx), 1)
        for a, p in zip(accum, params):
            noisy  = a + torch.normal(0.0, dp_sigma * clip, a.shape,
                                      device=a.device)
            p.grad = noisy / B
        opt.step()

    def fit(self, X, Y, epochs=50, batch_size=4, weight_decay=1e-4,
            learning_rate=1e-3, silent=True, lam=0.3, clip=1.0):
        X_t    = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
        y_diag = torch.LongTensor([int(y[0]) for y in Y]).to(device)
        y_sev  = torch.LongTensor([max(int(y[1]), 0) for y in Y]).to(device)
        s_mask = torch.FloatTensor([float(y[2]) for y in Y]).to(device)

        n_pos = int(y_diag.sum().item())
        n_neg = len(Y) - n_pos
        pos_w = torch.tensor(
            float((n_neg / n_pos) ** 0.5) if n_pos and n_neg else 1.0
        ).to(device)

        self.model.to(device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        n = X_t.size(0)
        dp_pe = dp_sigma > 0 and dp_per_example

        for e in range(epochs):
            perm, tot = torch.randperm(n), 0.0
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]

                if dp_pe:
                    _dp_step(self, X_t, y_diag, y_sev, s_mask, b.tolist(),
                             opt, lam, clip)
                    continue

                opt.zero_grad()
                diag, sev = self.model(X_t[b])
                loss, ld, ls = multitask_loss(diag, sev, y_diag[b], y_sev[b],
                                              s_mask[b], pos_w, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=clip)
                if dp_sigma > 0:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad += torch.normal(
                                0.0, dp_sigma * clip, p.grad.shape,
                                device=p.grad.device
                            )
                opt.step()
                tot += loss.item()
            if not dp_pe:
                sched.step(tot)
            if not silent:
                print(f"  epoch {e+1}: loss={tot:.4f}")

    def predict(self, X, Y=None, batch_size=8, GET_ACT=False):
        """Returns (diag_prob [n], sev_level [n], sev_logits [n, K-1])."""
        self.model.to(device).eval()
        X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)

        dp, sl = [], []
        with torch.no_grad():
            for s in range(0, X_t.size(0), batch_size):
                diag, sev = self.model(X_t[s:s + batch_size])
                dp.append(torch.sigmoid(diag).squeeze(-1).cpu())
                sl.append(sev.cpu())
        dp = torch.cat(dp).numpy()
        sl = torch.cat(sl)
        return dp, coral_to_level(sl).numpy(), sl.numpy()

    wrapper_cls.fit     = fit
    wrapper_cls.predict = predict
    return wrapper_cls


# ═════════════════════════════════════════════════════════════════════════════
# 4. Severity-aware FedAvg
# ═════════════════════════════════════════════════════════════════════════════
SEV_PARAM_KEYS = ("sev_fc", "sev_bias")


def fedavg_multitask(param_lists, client_sizes, client_sev_counts, state_keys):
    """
    Standard FedAvg on the trunk, but the severity head is weighted by each
    client's count of LABELLED severity samples, not total samples.

    Why: MGH Pediatrics has 24 samples and 10 severity labels; Cedars-Sinai has
    78 and 41. Under plain FedAvg a client with few (or zero) severity labels
    still drags sev_fc / sev_bias toward its near-untrained values at full
    weight. Weighting the severity head separately is a small, defensible
    contribution and worth naming in the paper.
    """
    tot_n   = float(sum(client_sizes))
    tot_sev = float(sum(client_sev_counts))
    n_c     = len(param_lists)

    out = []
    for i, key in enumerate(state_keys):
        is_sev = any(k in key for k in SEV_PARAM_KEYS)
        if is_sev and tot_sev > 0:
            w = [c / tot_sev for c in client_sev_counts]
        else:
            w = [c / tot_n for c in client_sizes]
        out.append(sum(param_lists[c][i] * w[c] for c in range(n_c)))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. Metrics
# ═════════════════════════════════════════════════════════════════════════════
def severity_metrics(pred_level, true_level, mask, n_levels=4):
    """
    Spearman rho, quadratic-weighted kappa, MAE in levels, plus a
    majority-class baseline MAE so the number has context.

    Do NOT report severity accuracy on its own — with 58/26/24/10 it is
    dominated by the remission class and looks naive.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    import warnings

    m = np.asarray(mask).astype(bool)
    p, t = np.asarray(pred_level)[m], np.asarray(true_level)[m]
    if len(t) < 3 or len(np.unique(t)) < 2:
        return {"n": int(len(t)), "note": "too few labelled samples"}

    # Early rounds often predict a single level for everything, which makes
    # Spearman undefined. That is informative (the head has not learned yet),
    # not an error — report 0.0 rather than spamming warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(p, t).correlation
    qwk = cohen_kappa_score(t, p, weights="quadratic",
                            labels=list(range(n_levels)))
    mae = float(np.abs(p - t).mean())
    baseline = float(np.abs(np.full_like(t, np.bincount(t).argmax()) - t).mean())

    return {
        "n": int(len(t)),
        "spearman": float(rho) if rho == rho else 0.0,
        "qwk": float(qwk),
        "mae_levels": mae,
        "mae_majority_baseline": baseline,
        "confusion": confusion_matrix(t, p,
                                      labels=list(range(n_levels))).tolist(),
    }


def diagnosis_metrics(prob, y_true):
    from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                                 average_precision_score, recall_score)
    yp = (np.asarray(prob) > 0.5).astype(int)
    yt = np.asarray(y_true).astype(int)
    return {
        "mcc": float(matthews_corrcoef(yt, yp)),
        "auc": float(roc_auc_score(yt, prob)) if len(np.unique(yt)) > 1 else 0.5,
        "auprc": float(average_precision_score(yt, prob)),
        "sen": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "spe": float(recall_score(yt, yp, pos_label=0, zero_division=0)),
    }
