# Results summary

All numbers from the merged notebook. Where a CSV is missing, the values were
read from the notebook output (Kaggle Quick Save stores only the last-executed
cell's files).

## Headline — config A, 4 levels, region-normalised, 3 seeds

| metric | mean ± std | per seed (42 / 7 / 123) |
|---|---|---|
| severity QWK | 0.234 ± 0.053 | 0.307 / 0.213 / 0.182 |
| severity Spearman | 0.289 ± 0.065 | 0.380 / 0.257 / 0.230 |
| severity MAE | 0.780 ± 0.053 | 0.723 / 0.766 / 0.851 |
| majority baseline MAE | 0.828 | — |
| diagnosis MCC | 0.212 ± 0.072 | 0.261 / 0.266 / 0.110 |
| diagnosis AUC | 0.637 ± 0.034 | 0.674 / 0.643 / 0.593 |

A single seed was NOT reproducible — QWK ranged 0.18–0.31. Report seed-averaged
values only.

## Federated vs centralized (matched folds, seed, features, checkpointing)

| | MCC | AUC | QWK | Spearman | MAE |
|---|---|---|---|---|---|
| Federated | 0.261 | 0.674 | 0.307 | 0.380 | 0.723 |
| Centralized | 0.224 | 0.678 | 0.257 | 0.338 | 0.760 |

Federation costs no utility. Likely because averaging 4 models trained on ~50
samples each regularises, while 100 epochs on 200 samples overfits.

## Feature ablation (seed 42)

| config | MCC | QWK | Spearman |
|---|---|---|---|
| baseline | 0.276 | 0.229 | 0.338 |
| region z-score (A) | 0.261 | 0.307 | 0.380 |
| region flag (B) | 0.319 | 0.260 | 0.338 |
| both (C) | 0.212 | 0.299 | 0.414 |

Tissue identity helps diagnosis; removing tissue variation helps severity.
Opposite operations, so no single config wins both.

## Per-site (pooled across folds, consistent across seeds)

| site | n_sev | MCC | severity MAE | its baseline |
|---|---|---|---|---|
| Cedars-Sinai | 41 | 0.242 | 0.488 | 0.512 ✓ |
| Cincinnati | 39 | 0.436 | 1.103 | 0.692 ✗ |
| MGH | 28 | -0.095 | 0.571 | 0.500 ✗ |
| MGH Pediatrics | 10 | 0.258 | 0.700 | 0.600 ✗ |

Only the remission-heavy site beats its own baseline. Three of four clients are
remission-dominated, so FedAvg pulls the global model toward predicting
remission — good for Cedars-Sinai (68% remission), bad for Cincinnati whose
patients mostly are not.

## Non-IID severity distribution (row %)

| site | remission | mild | moderate | severe |
|---|---|---|---|---|
| Cedars-Sinai | 68.3 | 22.0 | 0.0 | 9.8 |
| Cincinnati | 20.5 | 12.8 | 51.3 | 15.4 |
| MGH | 53.6 | 42.9 | 3.6 | 0.0 |
| MGH Pediatrics | 70.0 | 0.0 | 30.0 | 0.0 |

## Severity XAI — 20 genes stable across 3 seeds

mean pairwise top-50 overlap = 0.573

`ABO CDKN2A CLDN1 COL8A2 DAO DDO F2 GSDMA IGHG1 IL2 MEFV MICA MT1A MT1H MT1M
MT1X NLRP12 PER3 PTGS2 ZPBP2`

- **Severity-specific:** four metallothioneins (MT1X/MT1A/MT1H/MT1M), plus the
  D-amino acid oxidase pair DDO/DAO. Zinc-binding stress response.
- **Diagnosis-specific:** ABO (diag rank 1), FUT2, FUT3 — the blood-group and
  secretor-status loci, among the best-replicated IBD susceptibility genes.

The two heads separate risk loci from activity markers without supervision.
All 20 genes confirmed present in the expression matrix (none zero-filled).

## Differential privacy — infeasible

| approach | params | epsilon | MCC | outcome |
|---|---|---|---|---|
| client-level (CAGI) | 708k | 1211 | -0.05 | destroyed |
| example-level, sigma=2.0 | 708k | 8.3 | 0.02 | destroyed |
| example-level, sigma=0.6 | 708k | 747 | 0.06 | destroyed |
| mean pooling, sigma=0 | 1.5k | — | 0.05 | broke without noise |
| attention pooling, sigma=0 | 1.5k | — | 0.01 | broke without noise |
| top-100 genes, no DP | 102k | — | 0.191 | **utility held** |
| top-100 genes, sigma=1.5 | 102k | 27.8 | 0.003 | destroyed |

The last pair is the decisive one. Gene subsetting preserved utility without
noise (0.191 vs 0.212 full gene set), so the DP collapse is not a capacity
problem — it is the ~50-patients-per-site cohort size. Noise norm scales with
sqrt(d); at any d large enough to fit the task, it exceeds the gradient signal
50 patients can produce.

## Cohort

251 samples / 90 participants / 4 sites. 118 with SES-CD severity (CD only).
Severity levels 58 / 26 / 25 / 10. 657 of 691 genes present in the expression
matrix; 34 zero-filled.
