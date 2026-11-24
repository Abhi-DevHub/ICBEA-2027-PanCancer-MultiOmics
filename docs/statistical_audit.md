# ICBEA 2027 — Statistical & Scientific Audit Report

**Date:** 2026-11-24
**Scope:** verification of the revision experiments (`revision_*.csv`) prior to
editing the camera-ready paper.
**Principle applied:** no result was altered to obtain significance; where the
evidence is weaker than the original narrative, the narrative is corrected.

---

## Executive summary

| # | Audit item | Outcome |
|---|---|---|
| 1 | CI vs p-value inconsistency (XGB vs LR macro-F1) | **CONFIRMED — was a defect in my code. Fixed.** |
| 2 | Nadeau–Bengio implementation | **Correct** (variance, df, formula verified) |
| 3 | Imputation / scaling fitted per training fold | **Scaling: correct. Imputation: pre-split but a verified no-op.** |
| 4 | 32 per-class metrics | **Verified** (supports sum to 6,651; averages reconstruct exactly) |
| 5 | Uncertainty for modality ablations | **Now computed** (25 paired folds, NB-corrected) |
| 6 | Does removing mRNA genuinely improve macro-F1? | **NO — not significant (p = 0.507). Claim withdrawn.** |
| 7 | CellMarker / hypergeometric / BH | **Verified** (matches `scipy.hypergeom`, BH matches reference) |
| 8 | Every revised metric vs CSV | **Verified** |

---

## 1. The CI / p-value inconsistency (critical)

### What was wrong

The reported row was:

```
XGBoost vs LR, macro-F1:  Δ = +0.0176,  95% CI [0.0088, 0.0264],  p = 0.069
```

A 95% CI excluding zero is inconsistent with p > 0.05. **The two numbers came
from different variance assumptions:**

| Quantity | Variance used | Result |
|---|---|---|
| 95% CI (as reported) | **naive** `s_d / √J` (assumes independent folds) | [0.0088, 0.0264] — excludes 0 |
| p-value (as reported) | **Nadeau–Bengio** `√((1/J + n_test/n_train)·s_d²)` | 0.069 |

The CI ignored fold dependence, so it was too narrow. That is a defect in the
revision script, not in the experiment.

### Correction

Both quantities must use the **same** (Nadeau–Bengio) standard error.
With J = 25, df = 24, n_test/n_train = 1/4:

| Comparison | Metric | Mean Δ | 95% CI (NB) | d | p (NB, two-sided) | Significant |
|---|---|---|---|---|---|---|
| vs SVM | Weighted precision | +0.0060 | [0.0015, 0.0105] | 1.48 | 0.011 | yes |
| vs SVM | Weighted F1 | +0.0060 | [0.0017, 0.0103] | 1.55 | **0.008** | yes |
| vs SVM | Accuracy | +0.0061 | [0.0020, 0.0102] | 1.64 | 0.006 | yes |
| vs SVM | Cohen's κ | +0.0064 | [0.0021, 0.0107] | 1.64 | 0.006 | yes |
| vs SVM | Macro-F1 | +0.0070 | [−0.0145, 0.0283] | 0.36 | **0.509** | **no** |
| vs LR | Weighted precision | +0.0079 | [0.0030, 0.0128] | 1.78 | 0.003 | yes |
| vs LR | Weighted F1 | +0.0077 | [0.0031, 0.0123] | 1.88 | **0.002** | yes |
| vs LR | Accuracy | +0.0076 | [0.0031, 0.0120] | 1.89 | 0.002 | yes |
| vs LR | Cohen's κ | +0.0080 | [0.0033, 0.0126] | 1.90 | 0.002 | yes |
| vs LR | Macro-F1 | +0.0176 | [−0.0060, 0.0413] | 0.83 | **0.137** | **no** |

**All intervals now agree with their p-values** (verified programmatically:
`CI_agrees_with_p = YES` for all 10 rows).

---

## 2. Nadeau–Bengio implementation

Implemented as:

```
SE_NB  = sqrt( (1/J + n_test/n_train) · s_d² )
t      = mean_d / SE_NB
df     = J − 1
```

- `J = 25` repetitions — correct.
- `n_test/n_train = 1/4` for 5-fold — correct (test 1/5, train 4/5).
- `s_d` uses `ddof = 1` (unbiased) — correct.
- Inflation factor `1/25 + 0.25 = 0.29`, i.e. SE inflated by ×1.856 — correct.

Verified. The only defect was that the CI did not use it (now fixed).

---

## 3. Preprocessing and fold-fitting (reviewer comment 1)

| Operation | Where fitted | Verdict |
|---|---|---|
| **StandardScaler (main CV)** | `Pipeline([("scaler", StandardScaler()), ("clf", …)])` — refit per fold | ✅ correct |
| **StandardScaler (25 paired folds)** | `StandardScaler().fit(X_train.iloc[tr])` | ✅ correct |
| **StandardScaler (ablation CV)** | `StandardScaler().fit(Xtr.iloc[tr])` | ✅ correct |
| **StandardScaler (holdout)** | fit on `X_train`, applied to `X_test` | ✅ correct |
| **Column-mean imputation** | `common.py:99` — `fillna(df.mean())` on the **full** matrix, before splitting | ⚠️ pre-split |

**Imputation assessment.** The step runs before the split, which in principle is
leakage. However, we verified the raw matrices contain **zero missing values**
in all four modalities (mRNA, miRNA, CNV, Methylation — 0 NaNs each). The
operation therefore changes no value and cannot transfer information across the
split.

**How to describe this honestly in the paper:** state that imputation is a
defensive step, that the published matrices contain no missing values, and
therefore that no imputation-derived information crosses the split boundary. Do
not claim fold-wise imputation was performed — it was not.

**Optional hardening:** move imputation inside the fold. Results would be
identical (no-op), but it removes the criticism entirely.

---

## 4. Per-class metrics (32 classes) — verified

- 32 rows, supports sum to **6,651** = training partition size ✅
- Macro-F1 recomputed from the per-class rows: **0.9461** = reported 0.9461 ✅
- Weighted-F1 recomputed: **0.9720** = reported 0.9720 ✅
- Classes below F1 = 0.90: **OV (n=7, F1=0.600)**, **READ (n=73, F1=0.677)**,
  **UCS (n=44, F1=0.897)**.

⚠️ **Note for the paper text:** OV has **7** samples in the *training partition*
(9 in the full cohort) and READ has **73** (91 in the full cohort). Any sentence
quoting n = 9 / 91 must specify "full cohort", or be updated to 7 / 73 for the
train-split evaluation.

---

## 5–6. Leave-one-modality-out ablation with uncertainty (critical)

25 paired folds (5 repeats × 5 folds) inside the training partition, paired
against the full model, Nadeau–Bengio corrected.

### Accuracy (full − configuration)

| Comparison | Mean Δ | 95% CI | d | p (NB) | Significant |
|---|---|---|---|---|---|
| full − without mRNA | +0.0060 | [0.0021, 0.0100] | 1.69 | 0.004 | yes |
| full − without miRNA | +0.0072 | [0.0043, 0.0100] | 2.79 | 3×10⁻⁵ | yes |
| full − without CNV | −0.0010 | [−0.0040, 0.0020] | −0.37 | 0.501 | no |
| full − without Methylation | +0.0057 | [0.0019, 0.0096] | 1.66 | 0.005 | yes |
| full − mRNA-only | +0.0173 | [0.0141, 0.0204] | 6.07 | <10⁻⁵ | yes |

### Macro-F1 (full − configuration)

| Comparison | Mean Δ | 95% CI | d | p (NB) | Significant |
|---|---|---|---|---|---|
| full − without mRNA | +0.0064 | [−0.0131, 0.0258] | 0.36 | **0.507** | **no** |
| full − without miRNA | +0.0125 | [−0.0039, 0.0289] | 0.85 | **0.128** | **no** |
| full − without CNV | −0.0016 | [−0.0185, 0.0154] | −0.10 | 0.851 | no |
| full − without Methylation | +0.0093 | [−0.0011, 0.0198] | 0.99 | **0.078** | **no** |
| full − mRNA-only | +0.0368 | [0.0249, 0.0487] | 3.43 | <10⁻⁵ | **yes** |

### Corrected interpretation

1. **Removing any single modality does not significantly reduce macro-F1.**
   Only using mRNA *alone* does (−3.7 pts, p < 10⁻⁵).
2. Removing **miRNA**, **mRNA** or **methylation** significantly reduces
   **accuracy** (p ≤ 0.005); removing **CNV** does not.
3. **The earlier claim that "removing mRNA improves macro-F1" is NOT supported.**
   The point estimate was +0.006 with a CI spanning zero (p = 0.507). This claim
   is withdrawn; the defensible statement is that removing mRNA *did not
   significantly degrade* macro-F1.
4. **mRNA-only is significantly worse than full multi-omics** on both metrics —
   this is the strongest and cleanest result in the ablation.

> ⚠️ Wilcoxon p-values in the same table are much smaller than the
> Nadeau–Bengio values (e.g. 0.027 vs 0.507). This is expected: the plain
> Wilcoxon assumes independent folds and is anti-conservative under repeated CV.
> **Only the Nadeau–Bengio values should be reported.**

---

## 7. ORA implementation — verified

| Component | Check | Result |
|---|---|---|
| Hypergeometric `P(X ≥ k)` | Compared with `scipy.stats.hypergeom.sf` over 8 random parameter sets | **Exact match** |
| Benjamini–Hochberg | Compared with an independent reference implementation (40 p-values) | **Max abs diff = 0.0** |
| BH monotonicity / cap at 1 | Programmatic check | **Correct** |
| Background | Restricted to the 3,217 measured mRNA genes | Confirmed by script output |
| Zero-overlap sets | Excluded before testing | Confirmed in code |
| Gene-set parsing | Grouped by `cell_name`, `NA` excluded, exact symbol match | Confirmed in code |

One caveat: the ORA script could only be smoke-tested with a synthetic marker
file (the real CellMarker export is not in the repository). The statistics are
verified; a full re-run with the real CellMarker file is still required to
regenerate the paper's dot plot.

---

## 8. Metric cross-check against CSVs

All values in the replacement tables were read directly from:

`revision_5fold_cv_trainonly.csv`, `revision_per_class_trainonly.csv`,
`revision_paired_xgb_svm.csv`, `revision_significance_corrected.csv`,
`revision_ablation_leave_one_out.csv`, `revision_ablation_uncertainty.csv`.

Original artefacts (`5fold_cv_results.csv`, `per_class_report.csv`,
`wilcoxon_results.csv`, `ablation_comparison.csv`) were **not modified**.

---

## Claims: supported / unsupported / require qualification

### ✅ Supported

- XGBoost has the highest accuracy, weighted F1, weighted precision and κ under
  CV within the training partition.
- XGBoost is **significantly better than both SVM and Logistic Regression** on
  weighted precision, weighted F1, accuracy and κ (NB p ≤ 0.011).
- **mRNA-only is significantly worse** than full multi-omics (accuracy
  −1.73 pts, macro-F1 −3.68 pts, p < 10⁻⁵).
- Removing miRNA or methylation significantly reduces **accuracy**.
- Global feature importance does not equal modality-level necessity.

### ❌ Unsupported (must be removed or rewritten)

- "XGBoost significantly outperforms … on macro-F1." → **Not supported**
  (vs SVM p = 0.509; vs LR p = 0.137).
- "Removing mRNA improves macro-F1." → **Not supported** (p = 0.507).
- "mRNA is the essential/dominant modality." → **Contradicted.**
- "All four modalities are individually necessary." → **Contradicted**
  (CNV removal: p = 0.501 accuracy, p = 0.851 macro-F1).

### ⚠️ Requires qualification

- Fold independence: the 25 folds are not independent; only NB-corrected values
  are reportable.
- Small classes: OV n = 7 (train) / 9 (full cohort); macro-F1 variance is driven
  by these.
- Cutoff sensitivity: ORA significance holds only at N = 150.
- Causal language: SHAP/ORA are associational, not causal.
- Literature comparison (CMGL/MOGONET) is *reported* results, not a controlled
  head-to-head run.

---

## Remaining work before submission

| Priority | Item |
|---|---|
| Critical | Paste corrected tables (CI/p now consistent) into Overleaf |
| Critical | Remove the two unsupported claims from abstract/conclusion |
| High | Reframe the mRNA finding as "importance ≠ necessity" (not "improves") |
| High | Add `\usepackage{multirow}` |
| High | Specify cohort scope wherever n = 9 / n = 91 appears |
| Medium | Re-run ORA with the real CellMarker file to regenerate Fig. 2 |
| Medium | Compile the full IEEE paper and check page limit |
| Low | Optionally move imputation inside folds (results unchanged) |
