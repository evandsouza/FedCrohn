"""
hmp2_loader.py — HMP2 severity-aware data loader for the FedCrohn GAT pipeline.

Produces data in EXACTLY the shape the existing model expects:
    X : [n_samples, 691, genesize]   (numGenes stays 691, adj_cache.pkl unchanged)
    Y : list of (y_diag, y_sev, sev_mask) tuples

Severity target: SES-CD (endoscopic), co-recorded with the biopsy the RNA
came from — zero temporal offset, unlike HBI which lives on stool-visit rows.

Usage on Kaggle:
    from hmp2_loader import build_hmp2_dataset
    data = build_hmp2_dataset(
        meta_path  = "/kaggle/input/.../hmp2_metadata_20180820.csv",
        counts_path= "/kaggle/input/.../host_tx_counts.tsv",
        geneList   = geneList,            # your existing sorted 691-gene list
    )
"""

import numpy as np
import pandas as pd

# ── SES-CD clinical cutoffs ──────────────────────────────────────────────────
# 4-level: standard trial thresholds. Level 3 has only n=10 in this cohort and
# level 2 is almost never predicted correctly (2/24), so the model effectively
# learns a 2-class split wearing 4 labels.
# 3-level: merges moderate+severe into "active disease". Gives 58/26/34, which
# is better balanced, and merges exactly the classes the model cannot separate.
SESCD_BINS_4   = [-np.inf, 2, 6, 15, np.inf]
SESCD_LABELS_4 = [0, 1, 2, 3]
SESCD_BINS_3   = [-np.inf, 2, 6, np.inf]
SESCD_LABELS_3 = [0, 1, 2]

# Defaults; overridden per call via n_levels=
SESCD_BINS   = SESCD_BINS_4
SESCD_LABELS = SESCD_LABELS_4
N_SEV_LEVELS = 4

SEV_COL  = "SES-CD Score"
ID_COL   = "External ID"
PART_COL = "Participant ID"
SITE_COL = "site_name"


# ═════════════════════════════════════════════════════════════════════════════
# 1. Metadata → labels
# ═════════════════════════════════════════════════════════════════════════════
def load_hmp2_labels(meta_path, merge_emory=True, verbose=True, n_levels=4):
    """
    Returns one row per host_transcriptomics sample with:
        External ID, Participant ID, site, diagnosis,
        y_diag   : 1 if CD else 0
        y_sev    : ordinal SES-CD level, -1 if not applicable
        sev_mask : 1.0 only for CD samples with a recorded SES-CD

    CRITICAL: sev_mask is 0 for UC and nonIBD. SES-CD scores Crohn's lesions,
    so UC/nonIBD sitting near 0 means "instrument does not apply", NOT
    "severity zero". Encoding them as level 0 makes the severity head relearn
    the diagnosis head, and severity metrics become meaningless.
    """
    df = pd.read_csv(meta_path, low_memory=False)
    tx = df[df.data_type == "host_transcriptomics"].copy()

    assert tx[ID_COL].is_unique, "External ID not unique — join key is unsafe"

    tx["y_diag"] = (tx["diagnosis"] == "CD").astype(int)

    bins   = SESCD_BINS_3   if n_levels == 3 else SESCD_BINS_4
    labels = SESCD_LABELS_3 if n_levels == 3 else SESCD_LABELS_4
    lvl = pd.cut(tx[SEV_COL], bins=bins, labels=labels).astype("float")

    is_cd_labelled = (tx["diagnosis"] == "CD") & tx[SEV_COL].notna()
    tx["y_sev"]    = np.where(is_cd_labelled, lvl, -1)
    tx["y_sev"]    = tx["y_sev"].fillna(-1).astype(int)
    tx["sev_mask"] = is_cd_labelled.astype(float)

    tx["site"] = tx[SITE_COL]
    if merge_emory:
        # Emory contributes 9 samples / 3 severity-labelled — too thin to be a
        # standalone FL client. Set merge_emory=False to keep it as a straggler.
        tx["site"] = tx["site"].replace({"Emory": "MGH"})

    # Ileum (small intestine) vs colorectal (large intestine): ~50/50 in HMP2,
    # very different baseline expression. Independent of diagnosis and severity
    # (chi2 p~0.47 both), so not a confound — but it is variance worth removing.
    tx["region"] = np.where(
        tx["biopsy_location"].isin(["Ileum", "Terminal ileum"]),
        "ileum", "colorectal")

    keep = [ID_COL, PART_COL, "site", "diagnosis", SEV_COL, "region",
            "y_diag", "y_sev", "sev_mask"]
    out = tx[keep].reset_index(drop=True)

    if verbose:
        print(f"host_transcriptomics samples : {len(out)}")
        print(f"  diagnosis  CD/UC/nonIBD    : "
              f"{(out.diagnosis=='CD').sum()}/"
              f"{(out.diagnosis=='UC').sum()}/"
              f"{(out.diagnosis=='nonIBD').sum()}")
        print(f"  severity-labelled (CD only): {int(out.sev_mask.sum())}"
              f"  across {out[out.sev_mask==1][PART_COL].nunique()} participants")
        print("  severity level counts      :",
              out[out.sev_mask == 1].y_sev.value_counts().sort_index().to_dict())
        print("\n  per-site (all samples / severity-labelled):")
        for s, g in out.groupby("site"):
            print(f"    {s:16s} {len(g):4d} / {int(g.sev_mask.sum()):3d}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. Expression matrix → 691-gene subset
# ═════════════════════════════════════════════════════════════════════════════
def load_expression(counts_path, geneList, sample_ids, verbose=True):
    """
    Reads host_tx_counts.tsv (genes x samples) and returns a
    [n_samples, 691] raw-count matrix aligned to `sample_ids` and `geneList`.

    Genes in geneList absent from the expression matrix are filled with zeros,
    so the 691-node graph and adj_cache.pkl stay valid untouched.
    """
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    counts.index = counts.index.astype(str)

    # HMP2 ships HGNC symbols, but handle Ensembl defensively.
    if counts.index.str.startswith("ENSG").mean() > 0.5:
        raise ValueError(
            "Expression matrix is keyed by Ensembl gene IDs. Map them to HGNC "
            "symbols first (pyensembl or a static ENSG->symbol table), then "
            "re-run. geneList uses symbols."
        )

    # Collapse duplicate symbols by summing (multiple transcripts -> one gene).
    if counts.index.duplicated().any():
        counts = counts.groupby(level=0).sum()

    kept    = [s for s in sample_ids if s in counts.columns]
    dropped = [s for s in sample_ids if s not in counts.columns]

    if not kept:
        raise KeyError(
            "No sample IDs matched columns in the counts file. Check that "
            "column headers are External IDs (e.g. 'CSM5FZ1F')."
        )
    if len(kept) < 0.5 * len(sample_ids):
        raise KeyError(
            f"Only {len(kept)}/{len(sample_ids)} samples matched — likely the "
            f"wrong join key. Counts columns look like: "
            f"{list(counts.columns[:3])}"
        )

    present = [g for g in geneList if g in counts.index]
    if verbose:
        print(f"\nexpression matrix : {counts.shape[0]} genes x {counts.shape[1]} samples")
        print(f"  geneList coverage: {len(present)}/{len(geneList)} "
              f"({100*len(present)/len(geneList):.1f}%)")
        if len(present) < 0.8 * len(geneList):
            print("  WARNING: low coverage — check gene identifier format")
        if dropped:
            print(f"  dropped {len(dropped)} sample(s) with metadata but no "
                  f"expression column: {dropped}")

    mat = pd.DataFrame(0.0, index=geneList, columns=kept)
    mat.loc[present, :] = counts.loc[present, kept].values
    return mat.T.values.astype(np.float32), kept    # [n_kept, 691], ids


# ═════════════════════════════════════════════════════════════════════════════
# 3. Raw counts → node features [n, 691, genesize]
# ═════════════════════════════════════════════════════════════════════════════
def make_node_features(raw_counts, train_idx=None, use_zscore=True,
                       region=None, region_zscore=False, region_flag=False):
    """
    Builds per-gene node feature vectors.

      dim 0 : log2(CPM + 1)                  — leakage-free, per-sample
      dim 1 : within-sample rank percentile  — leakage-free, per-sample
      dim 2 : per-gene z-score               — fit on train_idx ONLY
      dim 3 : tissue-region indicator        — only if region_flag=True

    Pass train_idx as the client's own training indices. Fitting the z-score
    on all samples leaks test distribution into training.

    HMP2 biopsies are ~50/50 ileum vs colorectal — two tissues with very
    different baseline expression. That is independent of diagnosis and
    severity (chi2 p~0.47 for both), so it is not a confound, but it is
    variance the model must otherwise absorb.

      region_zscore=True  z-score each gene WITHIN its tissue region, so
                          gene-level tissue differences are removed rather
                          than left for the GAT to model.
      region_flag=True    append a binary ileum indicator so the model can
                          still condition on tissue where it matters.

    `region` is an array of labels (e.g. 'ileum'/'colorectal') aligned to
    raw_counts rows. Required for either region option.

    Returns [n_samples, n_genes, genesize].
    """
    lib = raw_counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    cpm     = raw_counts / lib * 1e6
    log_cpm = np.log2(cpm + 1.0)

    order = np.argsort(np.argsort(log_cpm, axis=1), axis=1)
    rank  = order / max(log_cpm.shape[1] - 1, 1)

    feats = [log_cpm, rank]

    if use_zscore:
        idx = np.arange(raw_counts.shape[0]) if train_idx is None else np.asarray(train_idx)
        z   = np.zeros_like(log_cpm)

        if region_zscore and region is not None:
            reg = np.asarray(region)
            for r in np.unique(reg):
                rows    = np.where(reg == r)[0]
                tr_rows = np.intersect1d(rows, idx)
                # too few training rows in this region to estimate stats
                src = tr_rows if len(tr_rows) >= 5 else idx
                mu  = log_cpm[src].mean(axis=0, keepdims=True)
                sd  = log_cpm[src].std(axis=0, keepdims=True)
                sd[sd < 1e-8] = 1.0
                z[rows] = (log_cpm[rows] - mu) / sd
        else:
            mu = log_cpm[idx].mean(axis=0, keepdims=True)
            sd = log_cpm[idx].std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            z = (log_cpm - mu) / sd

        feats.append(z)

    if region_flag and region is not None:
        flag = (np.asarray(region) == "ileum").astype(np.float32)
        feats.append(np.repeat(flag[:, None], log_cpm.shape[1], axis=1))

    return np.stack(feats, axis=-1).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Federated partition + leakage-safe CV
# ═════════════════════════════════════════════════════════════════════════════
def partition_by_site(labels_df):
    """
    Real multi-centre partition. Participants never span sites, so grouping by
    site automatically prevents participant leakage across clients.
    Returns {site_name: np.array(row_indices)}.
    """
    return {s: g.index.values for s, g in labels_df.groupby("site")}


def participant_grouped_folds(labels_df, idx, n_folds=5, seed=42):
    """
    Within-client CV split by Participant ID, NOT by sample.

    189 samples come from 67 participants — a naive sample-level split puts
    repeat visits of the same person in both train and test, and inflates
    every metric you report.
    """
    rng   = np.random.default_rng(seed)
    parts = labels_df.loc[idx, PART_COL].unique()
    rng.shuffle(parts)
    buckets = np.array_split(parts, n_folds)

    folds = []
    for b in buckets:
        te = labels_df.loc[idx][labels_df.loc[idx, PART_COL].isin(b)].index.values
        tr = np.setdiff1d(idx, te)
        folds.append((tr, te))
    return folds


# ═════════════════════════════════════════════════════════════════════════════
# 5. Top-level builder
# ═════════════════════════════════════════════════════════════════════════════
def build_hmp2_dataset(meta_path, counts_path, geneList,
                       merge_emory=True, verbose=True, n_levels=4):
    labels    = load_hmp2_labels(meta_path, merge_emory=merge_emory,
                                 verbose=verbose, n_levels=n_levels)
    raw, kept = load_expression(counts_path, geneList,
                                labels[ID_COL].tolist(), verbose=verbose)

    # Realign labels to the samples that actually have expression, in the same
    # order as the rows of `raw`. Index is reset so client index arrays and
    # raw_counts rows refer to the same positions.
    labels = (labels.set_index(ID_COL)
                    .loc[kept]
                    .reset_index())
    assert len(labels) == raw.shape[0]

    if verbose:
        print(f"\nfinal cohort      : {len(labels)} samples")
        print(f"  severity-labelled: {int(labels.sev_mask.sum())}")
        print("  severity levels  :",
              labels[labels.sev_mask == 1].y_sev.value_counts().sort_index().to_dict())

    Y = list(zip(labels.y_diag.values,
                 labels.y_sev.values,
                 labels.sev_mask.values))

    clients = partition_by_site(labels)
    if verbose:
        print("\nfederated clients:")
        for s, ix in clients.items():
            print(f"  {s:16s} n={len(ix):4d}  severity-labelled="
                  f"{int(labels.loc[ix,'sev_mask'].sum()):3d}")

    return {"labels": labels, "raw_counts": raw, "Y": Y,
            "clients": clients, "geneList": geneList,
            "n_sev_levels": n_levels,
            "region": labels["region"].values}


# ═════════════════════════════════════════════════════════════════════════════
# 6. Synthetic dry-run — validate shapes today, without the tsv
# ═════════════════════════════════════════════════════════════════════════════
def synthetic_dryrun(meta_path, geneList, seed=42):
    """
    Runs the full label pipeline against REAL metadata but FAKE expression.
    Use this to shake out tensor shapes, masked-loss NaNs and FedAvg before
    you touch the real counts file.
    """
    rng    = np.random.default_rng(seed)
    labels = load_hmp2_labels(meta_path, verbose=True)
    raw    = rng.negative_binomial(5, 0.3,
                                   size=(len(labels), len(geneList))).astype(np.float32)

    tr = np.arange(len(labels))[: int(0.8 * len(labels))]
    X  = make_node_features(raw, train_idx=tr)

    print(f"\nX shape       : {X.shape}   -> genesize={X.shape[2]}, numGenes={X.shape[1]}")
    print(f"labelled sev  : {int(labels.sev_mask.sum())}")
    print(f"any NaN in X  : {np.isnan(X).any()}")
    return X, labels
