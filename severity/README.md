# Severity-Aware Extension

Adds SES-CD **severity** prediction alongside the existing Crohn's diagnosis,
trained federatedly across four real hospitals.

Nothing in `sources/` is modified. All changes are runtime monkey-patches
applied from the notebook, so the original FedCrohn pipeline is untouched and
still runnable.
## Quick start (Kaggle — no downloads)

Everything is already set up as a Kaggle dataset, GPU included:

1. Open [`bhanavi1231/hmp2-severity`](https://www.kaggle.com/datasets/bhanavi1231/hmp2-severity)
2. **New Notebook** (or Copy & Edit an existing one) — the dataset attaches automatically
3. Upload `fedcrohn_severity.ipynb` from this folder (File → Import Notebook)
4. Settings → Accelerator → **GPU T4 x2**, Internet → **Off**
5. Run setup cells 1-7, then any experiment cell

The dataset already contains `sources/`, `marshalledP3/`, the HMP2 files, and
the four modules — nothing to download or configure.

Mount path is `/kaggle/input/datasets/bhanavi1231/hmp2-severity`, which is the
CONFIG default.

---
## Why the dataset changed

CAGI2/3/4 has whole-exome sequencing but **no severity labels**, so a severity
head cannot be trained on it. HMP2 (ibdmdb.org) has host transcriptomics from
gut biopsies with SES-CD endoscopic scores recorded at the same appointment the
tissue came from.

One trap worth knowing: HMP2's `hbi`, `sccai` and `fecalcal` columns are empty
on host_transcriptomics rows — those are recorded at stool-collection visits.
`SES-CD Score` is the one co-recorded with biopsies, and it is what this
extension uses.

**Cohort:** 251 samples / 90 participants / 4 sites, of which 118 have SES-CD
(CD patients only). Severity levels 58 / 26 / 25 / 10.

## Architecture change

The graph is unchanged — same 691 genes, same STRING adjacency, same
`adj_cache.pkl`, same two GAT layers. Only the output end differs:

```
before:  GAT -> flatten -> classifier -> diagnosis
after:   GAT -> flatten -> shared trunk -+-> diagnosis
                                         +-> severity (CORAL, 4 ordered levels)
```

**CORAL rather than 4-way softmax** because severity levels are ordered:
predicting "severe" for a remission patient should cost more than predicting
"mild". One shared weight vector plus K-1 monotone biases keeps the cumulative
probabilities consistent, which softmax does not guarantee.

**Masked loss** because only 118 of 251 patients have a severity label. UC and
non-IBD patients contribute nothing to the severity term. Encoding them as
level 0 would make the severity head relearn the diagnosis head — severity
metrics would look good and mean nothing.

## Files

| file | contents |
|---|---|
| `hmp2_loader.py` | SES-CD labels, CD-only masking, 691-gene subset, site partition, region-aware node features |
| `severity_model.py` | CORAL head, masked multi-task loss, prior init, severity-weighted FedAvg, DP-SGD with per-example clipping, RDP accountant |
| `run_severity_fl.py` | federated loop, participant-grouped CV, per-site evaluation, gene subsetting, centralized baseline |
| `severity_xai.py` | gradient attribution contrasting the two heads |
| `fedcrohn_severity.ipynb` | **main notebook** — CONFIG block + all 9 experiments, runnable |
| `notebook_with_stored_outputs.ipynb` | archive: original Kaggle runs with outputs preserved |

## Running it

Requires the `hmp2-severity` Kaggle dataset (or equivalent local layout):

```
marshalledP3/{adj_cache.pkl, totGeneSet.m.min0}
phenopediaCrohnGenes/CrohnGenes.txt
sources/
host_tx_counts.tsv               # from ibdmdb.org, ~37 MB, not in this repo
hmp2_metadata_2018-08-20.csv
```

Open `fedcrohn_severity.ipynb`:

1. Edit the **CONFIG** cell (`PROJECT_PATH`, and any experiment flags).
2. Run **Setup** cells 1–7 in order.
3. Run **one** experiment cell.

Every CONFIG change needs a kernel restart — Cell 6 wraps `__init__`, so a
second execution in the same session double-wraps it. The experiment cells
that need a non-default CONFIG start with an `assert` that fails loudly rather
than running the wrong model.

| experiment | CONFIG |
|---|---|
| 1. Main result, 3 seeds | defaults |
| 2. Centralized baseline | defaults |
| 3. Feature ablation | defaults |
| 4. Non-IID site table | none (no training) |
| 5. Severity XAI | defaults |
| 6. Three-level severity | `N_SEV_LEVELS = 3` |
| 7. Gene subsetting | defaults |
| 8. Differential privacy | `DP_SIGMA = 1.5` |
| 9. Pooling ablation | `POOL_GENES = True` |

## Results

Full numbers in [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

**Severity prediction works, modestly.** QWK 0.234 ± 0.053, Spearman
0.289 ± 0.065, MAE 0.780 vs a 0.828 majority baseline. Seed-averaged over
42/7/123 — a single seed was not reproducible (QWK ranged 0.18–0.31).

**Federation costs no utility.** Matched against centralized training on the
same folds, seed, features and checkpoint protocol: federated 0.261 MCC /
0.307 QWK, centralized 0.224 / 0.257.

**Sites are genuinely Non-IID.** Cincinnati is 51% moderate-severity patients;
Cedars-Sinai is 0%. And it costs the minority site: only the remission-heavy
Cedars-Sinai beats its own per-site baseline, because three of four clients are
remission-dominated and FedAvg follows them.

**Gene attribution separates risk from activity.** 20 genes stable across three
seeds (0.573 mean top-50 overlap). The severity side is dominated by four
metallothioneins (MT1X/MT1A/MT1H/MT1M); the diagnosis side by ABO, FUT2 and
FUT3 — the blood-group and secretor-status loci that are among the
best-replicated IBD susceptibility genes. Neither grouping was supervised.

## Differential privacy does not work at this cohort size

Six attempts across two DP granularities and three model sizes. None achieved
both a meaningful budget (eps < 10) and utility above the majority baseline.

The decisive pair: gene subsetting to 100 genes cut the trunk from 708k to 102k
parameters and **preserved utility without noise** (MCC 0.191 vs 0.212), then
still collapsed under DP-SGD (MCC 0.003, eps 27.8). So the failure is not model
capacity — it is that noise norm scales with sqrt(d), and at any d large enough
to fit this task it exceeds the gradient signal ~50 patients per site can
produce.

Note the original pipeline's `clip_grad_norm_` is gradient clipping only, with
no noise and no per-example bound. It provides no privacy guarantee. The DP
implementation here adds per-example clipping and Gaussian noise, which is what
a valid (eps, delta) claim requires.

## Known limitations

- **Severity grading is effectively binary.** The model separates remission
  from active disease; it rarely identifies moderate (2/24) or severe (1/10).
  Only 25 moderate and 10 severe patients exist, and two of four sites have no
  moderate cases at all.
- **Diagnosis is a harder task here** than in the CAGI pipeline — CD vs
  (UC + non-IBD) on biopsy expression, rather than CD vs healthy. The 0.64 AUC
  is not directly comparable to the 0.94 reported on CAGI.
- Fold variance is large throughout (~21–25 labelled test patients per fold).
- Gene attribution is computed on a single fold per seed.

## Not done

lambda sweep, FedProx, attribution averaged across folds, 3-level severity as a
primary rather than robustness analysis.
