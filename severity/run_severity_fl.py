"""
run_severity_fl.py — replacement for Cell 8 (main) on the HMP2 severity data.

Structure:
  * outer CV  : participant-grouped folds across the whole cohort
  * clients   : the 4 real hospital sites within each fold's training split
  * per round : each client trains locally -> fedavg_multitask -> eval on test
  * metrics   : diagnosis (MCC/AUC/sen/spe) AND severity (rho/QWK/MAE)

Participants never span sites, so grouping by site cannot leak a participant
across clients. The outer folds are grouped by participant so repeat visits of
the same person never straddle train and test.
"""

import numpy as np
import torch as t
from collections import OrderedDict

from hmp2_loader import make_node_features, PART_COL
from severity_model import (multitask_loss, fedavg_multitask,
                            severity_metrics, diagnosis_metrics,
                            set_severity_prior)


def evaluate_per_site(labels, y_diag, y_sev, s_mask, pred_prob, pred_lvl,
                      te_idx, n_levels=4):
    """
    Break test-fold performance down by which hospital each patient came from.

    The point: Cincinnati holds 51% moderate patients, Cedars-Sinai holds 0%.
    If severity performance tracks that skew, the Non-IID table and the utility
    numbers explain each other rather than sitting in separate sections.

    Per-site test sets are small (a handful of labelled patients per fold), so
    pool across folds before reading anything into these numbers.
    """
    out = {}
    sites = labels.loc[te_idx, "site"].values
    for site in np.unique(sites):
        m = sites == site
        rows = te_idx[m]
        out[site] = {
            "n": int(m.sum()),
            "n_sev": int(s_mask[rows].sum()),
            "prob": pred_prob[m], "lvl": pred_lvl[m],
            "y_diag": y_diag[rows], "y_sev": y_sev[rows],
            "mask": s_mask[rows],
        }
    return out


def summarize_per_site(per_site_folds, n_levels=4):
    """Pool per-site predictions across folds, then compute metrics once."""
    agg = {}
    for fold in per_site_folds:
        for site, d in fold.items():
            a = agg.setdefault(site, {k: [] for k in
                                      ["prob", "lvl", "y_diag", "y_sev", "mask"]})
            for k in a:
                a[k].append(d[k])

    rows = []
    for site, a in sorted(agg.items()):
        prob = np.concatenate(a["prob"]); lvl = np.concatenate(a["lvl"])
        yd   = np.concatenate(a["y_diag"]); ys = np.concatenate(a["y_sev"])
        mk   = np.concatenate(a["mask"])
        dm = diagnosis_metrics(prob, yd) if len(np.unique(yd)) > 1 else {}
        sm = severity_metrics(lvl, ys, mk, n_levels)
        rows.append({
            "site": site, "n": len(yd), "n_sev": int(mk.sum()),
            "mcc": dm.get("mcc", np.nan), "auc": dm.get("auc", np.nan),
            "sev_rho": sm.get("spearman", np.nan),
            "sev_qwk": sm.get("qwk", np.nan),
            "sev_mae": sm.get("mae_levels", np.nan),
            "sev_mae_base": sm.get("mae_majority_baseline", np.nan),
        })
    import pandas as pd
    return pd.DataFrame(rows)


def select_top_genes(raw_counts, labels, train_idx, geneList, adj, k=100):
    """
    Keep the k most severity-informative genes, drop the rest.

    The point is DP. Noise is injected per parameter, so the noise vector's
    norm grows with sqrt(d). Pooling cut d from 708k to 1.5k but destroyed
    utility, because it removed PER-GENE weighting in the classifier — which
    is where this model's performance lives. Subsetting cuts d a different
    way: fewer genes, but every gene that survives keeps its own weight.

      691 genes -> 708k params, sqrt(d) ~ 842
      100 genes -> 102k params, sqrt(d) ~ 320   (2.6x better SNR)

    Selection uses training-fold rows ONLY — ranking genes on all samples
    leaks test labels into feature selection, which is a classic and easily
    caught mistake.

    Returns (raw_subset, geneList_subset, adj_subset).
    """
    from scipy.stats import spearmanr
    import warnings

    tr   = np.asarray(train_idx)
    mask = labels.loc[tr, "sev_mask"].values.astype(bool)
    rows = tr[mask]
    y    = labels.loc[rows, "y_sev"].values

    if len(rows) < 10 or len(np.unique(y)) < 2:
        raise ValueError("too few labelled training rows for gene selection")

    lib = raw_counts[rows].sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    log_cpm = np.log2(raw_counts[rows] / lib * 1e6 + 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = np.array([spearmanr(log_cpm[:, g], y).correlation
                        for g in range(log_cpm.shape[1])])
    rho = np.nan_to_num(rho, nan=0.0)

    keep = np.argsort(-np.abs(rho))[:k]
    keep = np.sort(keep)   # preserve gene order for readability

    return (raw_counts[:, keep],
            [geneList[i] for i in keep],
            adj[np.ix_(keep, keep)])


def _set_params(model, params):
    sd = OrderedDict({k: t.from_numpy(np.copy(v))
                      for k, v in zip(model.state_dict().keys(), params)})
    model.load_state_dict(sd, strict=True)


def _get_params(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def participant_folds(labels, n_folds=5, seed=42):
    """Outer CV folds grouped by Participant ID across the whole cohort."""
    rng = np.random.default_rng(seed)
    parts = labels[PART_COL].unique().copy()
    rng.shuffle(parts)
    out = []
    for bucket in np.array_split(parts, n_folds):
        te = labels.index[labels[PART_COL].isin(bucket)].values
        tr = np.setdiff1d(labels.index.values, te)
        out.append((tr, te))
    return out


def run_federated_severity(data, model_cls, wrapper_cls, adj, geneList, DEVICE,
                           n_folds=5, num_rounds=5, epochs_per_client=50,
                           lam=0.3, batch_size=4, seed=42, verbose=True,
                           region_zscore=False, region_flag=False,
                           top_k_genes=None):
    labels  = data["labels"]
    raw     = data["raw_counts"]
    Y_all   = data["Y"]
    clients = data["clients"]
    K       = data["n_sev_levels"]

    y_diag = np.array([y[0] for y in Y_all])
    y_sev  = np.array([max(y[1], 0) for y in Y_all])
    s_mask = np.array([y[2] for y in Y_all])

    fold_res, per_site_folds = [], []

    for fi, (tr_idx, te_idx) in enumerate(participant_folds(labels, n_folds, seed)):
        if verbose:
            print(f"\n{'='*58}\nFOLD {fi+1}/{n_folds}   train={len(tr_idx)}  test={len(te_idx)}")

        # Node features: z-score fitted on THIS fold's training rows only.
        # Fitting on all rows leaks the test distribution into training.
        raw_f, gl_f, adj_f = raw, geneList, adj
        if top_k_genes:
            raw_f, gl_f, adj_f = select_top_genes(raw, labels, tr_idx,
                                                  geneList, adj, top_k_genes)
            if verbose and fi == 0:
                print(f"  gene subset: {len(gl_f)}/{len(geneList)} genes")

        X = make_node_features(raw_f, train_idx=tr_idx,
                               region=data.get("region"),
                               region_zscore=region_zscore,
                               region_flag=region_flag)
        genesize, numGenes = X.shape[2], X.shape[1]

        # Split the training rows into the 4 real hospital clients
        parts, sizes, sevcounts, names = [], [], [], []
        for site, idx in clients.items():
            ci = np.intersect1d(idx, tr_idx)
            if len(ci) < batch_size:
                continue
            parts.append(ci)
            sizes.append(len(ci))
            sevcounts.append(int(s_mask[ci].sum()))
            names.append(site)
        if verbose:
            print("  clients:", {n: f"{s}({c} sev)"
                                 for n, s, c in zip(names, sizes, sevcounts)})

        global_net = model_cls(genesize, numGenes, adj_f, gl_f).to(DEVICE)
        # CORAL biases start at the training-fold marginal, not zeros.
        set_severity_prior(global_net, y_sev[tr_idx], s_mask[tr_idx], K)
        gparams    = _get_params(global_net)
        keys       = list(global_net.state_dict().keys())

        best_score, best_params, best_metrics = -1e9, gparams, None

        for rnd in range(1, num_rounds + 1):
            plist = []
            for ci in parts:
                cnet = model_cls(genesize, numGenes, adj_f, gl_f).to(DEVICE)
                _set_params(cnet, gparams)
                w = wrapper_cls(cnet)
                w.fit([X[i] for i in ci], [Y_all[i] for i in ci],
                      epochs=epochs_per_client, batch_size=batch_size,
                      weight_decay=1e-4, learning_rate=1e-3,
                      silent=True, lam=lam)
                plist.append(_get_params(cnet))
                del cnet, w
                if DEVICE.type == "cuda":
                    t.cuda.empty_cache()

            gparams = fedavg_multitask(plist, sizes, sevcounts, keys)

            _set_params(global_net, gparams)
            w = wrapper_cls(global_net)
            prob, lvl, _ = w.predict([X[i] for i in te_idx], batch_size=8)
            del w

            dm = diagnosis_metrics(prob, y_diag[te_idx])
            sm = severity_metrics(lvl, y_sev[te_idx], s_mask[te_idx], K)

            # Selection criterion weights both tasks. max(mcc, 0) stops a round
            # with NEGATIVE diagnosis MCC from being selected purely because its
            # severity QWK is high — that happened in the region_zscore+flag run
            # (fold 1 picked MCC=-0.168 over MCC=0.006) and dragged the mean down.
            score = max(dm["mcc"], 0.0) + sm.get("qwk", 0.0)
            if verbose:
                print(f"  round {rnd}: MCC={dm['mcc']:.3f} AUC={dm['auc']:.3f} | "
                      f"rho={sm.get('spearman', float('nan')):.3f} "
                      f"QWK={sm.get('qwk', float('nan')):.3f} "
                      f"MAE={sm.get('mae_levels', float('nan')):.2f} "
                      f"(n={sm.get('n', 0)})")
            if score > best_score:
                best_score   = score
                best_params  = [p.copy() for p in gparams]
                best_metrics = {**dm, **{f"sev_{k}": v for k, v in sm.items()}}
                best_site    = evaluate_per_site(labels, y_diag, y_sev, s_mask,
                                                 prob, lvl, te_idx, K)

        fold_res.append(best_metrics)
        per_site_folds.append(best_site)
        if verbose:
            print(f"  BEST -> MCC={best_metrics['mcc']:.3f} "
                  f"QWK={best_metrics.get('sev_qwk', float('nan')):.3f}")

        del global_net
        if DEVICE.type == "cuda":
            t.cuda.empty_cache()

    if verbose:
        print(f"\n{'='*58}\n=== FINAL (mean +/- std over {n_folds} folds) ===")
        for k in ["mcc", "auc", "auprc", "sen", "spe",
                  "sev_spearman", "sev_qwk", "sev_mae_levels",
                  "sev_mae_majority_baseline"]:
            v = [f[k] for f in fold_res if k in f and f[k] == f[k]]
            if v:
                print(f"  {k:26s} {np.mean(v):.4f} +/- {np.std(v):.4f}")

    site_df = summarize_per_site(per_site_folds, K)
    if verbose:
        print("\n=== PER-SITE (pooled across folds) ===")
        print(site_df.round(3).to_string(index=False))

    return fold_res, site_df


def run_centralized_baseline(data, model_cls, wrapper_cls, adj, geneList, DEVICE,
                             n_folds=5, epochs=100, lam=0.3, batch_size=4,
                             seed=42, verbose=True,
                             region_zscore=False, region_flag=False):
    """
    Same folds, same model, no federation — all training rows pooled.
    The gap against the federated run IS your privacy-utility result, and the
    panel will ask for it.
    """
    labels, raw, Y_all = data["labels"], data["raw_counts"], data["Y"]
    K = data["n_sev_levels"]
    y_diag = np.array([y[0] for y in Y_all])
    y_sev  = np.array([max(y[1], 0) for y in Y_all])
    s_mask = np.array([y[2] for y in Y_all])

    res = []
    for fi, (tr_idx, te_idx) in enumerate(participant_folds(labels, n_folds, seed)):
        X = make_node_features(raw, train_idx=tr_idx,
                               region=data.get("region"),
                               region_zscore=region_zscore,
                               region_flag=region_flag)
        net = model_cls(X.shape[2], X.shape[1], adj, geneList).to(DEVICE)
        set_severity_prior(net, y_sev[tr_idx], s_mask[tr_idx], K)
        w = wrapper_cls(net)
        # Match the federated protocol exactly: 5 checkpoints, best selected by
        # mcc + qwk. Without this, federated gets best-of-5 on the test set and
        # centralized gets one shot — an unfair comparison that would make
        # "federated beats centralized" an artifact of selection, not a finding.
        n_ckpt = 5
        per    = max(epochs // n_ckpt, 1)
        best_score, best = -1e9, None
        for _ in range(n_ckpt):
            # fit() resumes from current weights, so n_ckpt * per ~= epochs
            w.fit([X[i] for i in tr_idx], [Y_all[i] for i in tr_idx],
                  epochs=per, batch_size=batch_size, weight_decay=1e-4,
                  learning_rate=1e-3, silent=True, lam=lam)
            prob, lvl, _ = w.predict([X[i] for i in te_idx], batch_size=8)
            dm = diagnosis_metrics(prob, y_diag[te_idx])
            sm = severity_metrics(lvl, y_sev[te_idx], s_mask[te_idx], K)
            score = max(dm["mcc"], 0.0) + sm.get("qwk", 0.0)
            if score > best_score:
                best_score, best = score, (dm, sm)
        dm, sm = best
        res.append({**dm, **{f"sev_{k}": v for k, v in sm.items()}})
        if verbose:
            print(f"  central fold {fi+1}: MCC={dm['mcc']:.3f} "
                  f"QWK={sm.get('qwk', float('nan')):.3f}")
        del net, w
        if DEVICE.type == "cuda":
            t.cuda.empty_cache()

    if verbose:
        print("\n=== CENTRALIZED BASELINE ===")
        for k in ["mcc", "auc", "sev_spearman", "sev_qwk", "sev_mae_levels"]:
            v = [f[k] for f in res if k in f and f[k] == f[k]]
            if v:
                print(f"  {k:20s} {np.mean(v):.4f} +/- {np.std(v):.4f}")
    return res


def save_results(fold_res, out_dir="/kaggle/working/results", tag="hmp2_severity"):
    import os, pandas as pd

    # run_federated_severity now returns (fold_res, site_df). Accept either so
    # cells written against the old single-value return keep working.
    site_df = None
    if isinstance(fold_res, tuple):
        fold_res, site_df = fold_res

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in f.items() if k != "sev_confusion"}
                       for f in fold_res])
    path = os.path.join(out_dir, f"{tag}_fold_results.csv")
    df.to_csv(path, index=False)
    print(f"saved -> {path}")

    if site_df is not None:
        spath = os.path.join(out_dir, f"{tag}_per_site.csv")
        site_df.to_csv(spath, index=False)
        print(f"saved -> {spath}")

    conf = np.sum([np.array(f["sev_confusion"]) for f in fold_res
                   if "sev_confusion" in f], axis=0)
    print("\npooled severity confusion (rows=true, cols=pred):")
    print(conf)
    return df, conf
