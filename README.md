# Lineage-Associated Interpretable Pan-Cancer Tumor-Type Classification using Multi-Omics Data

Reference implementation for the ICBEA 2027 paper of the same title.

The project benchmarks four classical classifiers for 32-class pan-cancer
tumor-type classification on the MLOmics multi-omics benchmark, subjects the
comparison to paired statistical testing with a correction for repeated-CV fold
dependence, and then asks a question that feature-importance rankings cannot
answer on their own: **does a highly-ranked feature modality actually matter for
prediction?**

---

## 1. Overview

- **Task:** 32-class tumor-type classification from four harmonised molecular
  modalities (8,314 samples, 9,844 features).
- **Models:** XGBoost, linear SVM, Random Forest, Logistic Regression.
- **Evaluation:** stratified 5-fold cross-validation performed *inside* the 80%
  training partition; a 20% holdout is reserved for final evaluation.
- **Statistics:** 25 repeated-CV folds, paired comparisons corrected with the
  Nadeau–Bengio resampled test, with confidence intervals and effect sizes.
- **Interpretability:** SHAP feature ranking, background-constrained
  over-representation analysis, and leave-one-modality-out ablations.

## 2. Motivation

Multi-omics profiling is widely used for tumor characterization, but two
practical problems recur in published benchmarks:

1. **Evaluation protocol.** A single cross-validation run gives too few paired
   observations for a sound statistical comparison, and repeated-CV folds are
   not independent — treating them as independent inflates significance.
2. **Interpretation of importance.** A high SHAP or gain rank is frequently read
   as evidence that a feature, or an entire modality, is *necessary*. That
   inference is not valid on its own.

This repository addresses both: it reports a corrected repeated-CV comparison,
and it tests modality necessity directly with leave-one-modality-out ablations.

## 3. Dataset

**MLOmics: Cancer Multi-Omics Database for Machine Learning**
(Yang et al., *Scientific Data* 12:913, 2025), derived from TCGA.

- Paper: https://doi.org/10.1038/s41597-025-05235-x
- Code / database: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark

Cohort used here: **8,314 samples**, **32 tumor types**, class sizes from 9 to
757 samples (≈84× skew).

> Raw data is **not** redistributed. See [`data/README.md`](data/README.md).

## 4. Omics modalities

| Modality | Features | Role |
|---|---:|---|
| mRNA expression | 3,217 | Lineage/transcription signal; dominates importance rankings |
| microRNA (miRNA) | 383 | Post-transcriptional regulation |
| Copy-number variation (CNV) | 3,105 | Large-scale genomic gains and losses |
| DNA methylation | 3,139 | Epigenetic programming |
| **Total** | **9,844** | Concatenated feature space |

## 5. Preprocessing

1. Each modality matrix transposed so samples are rows.
2. Feature names suffixed by modality (`_mRNA`, `_miRNA`, `_CNV`, `_Methy`).
3. Inner join of the four matrices on TCGA sample identifiers → 8,314 samples.
4. Column-mean imputation applied as a defensive step.
5. Stratified **80/20** train/test split (`random_state=42`).
6. `StandardScaler` **fit on training data only**.

**On leakage.** Scaling is refitted independently inside every cross-validation
fold. All cross-validation is performed strictly within the 80% training
partition; the holdout is used only for the final evaluation. The published
matrices contain **no missing values** (verified across all four modalities), so
the imputation step changes no value and cannot transfer information across the
split boundary.

## 6. Methodology

1. Preprocess and concatenate the four modalities.
2. Fit and evaluate four classifiers on the holdout split.
3. Run 5-fold stratified CV **within the training partition** for each classifier.
4. Repeat the 5-fold split five times with different shuffles → 25 paired folds;
   compare XGBoost against **both** SVM and Logistic Regression on identical folds.
5. Correct the paired comparisons for fold dependence (Nadeau–Bengio).
6. Rank features by SHAP and by split-gain importance; compare the two.
7. Run ORA on the top-ranked mRNA genes against CellMarker cell types.
8. Run leave-one-modality-out ablations with paired uncertainty.

See [`docs/methodology.md`](docs/methodology.md) for detail.

## 7. Classifiers and hyperparameters

| Model | Configuration |
|---|---|
| XGBoost | `n_estimators=300` (200 in CV), `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, GPU `device='cuda'` with CPU fallback |
| Linear SVM | `SVC(kernel='linear')`, `random_state=42` |
| Random Forest | `n_estimators=300` (200 in CV), `random_state=42` |
| Logistic Regression | `max_iter=2000`, `solver='lbfgs'`, `random_state=42` |

Hyperparameters were **not** tuned. This is a deliberate choice: tuning only some
models would confound the comparison. Note that this is a fixed-protocol
benchmark, not a search for peak performance.

## 8. Five-fold cross-validation

Cross-validation is stratified (`StratifiedKFold`, shuffle, `random_state=42`)
and executed **within the 80% training partition** (6,651 samples). Fold
consistency is reported as `1 − σ/μ` of weighted precision. The 20% holdout
(1,663 samples) is untouched during CV and model selection.

## 9. Performance comparison

Stratified 5-fold CV within the training partition (mean ± standard deviation):

| Model | Accuracy | Macro-F1 | Cohen's κ |
|---|---:|---:|---:|
| **XGBoost** | **0.9725 ± 0.0029** | **0.9443 ± 0.0184** | **0.9712 ± 0.0033** |
| SVM | 0.9663 ± 0.0044 | 0.9405 ± 0.0182 | 0.9647 ± 0.0046 |
| Logistic Regression | 0.9626 ± 0.0056 | 0.9290 ± 0.0206 | 0.9608 ± 0.0058 |
| Random Forest | 0.9505 ± 0.0050 | 0.8936 ± 0.0066 | 0.9481 ± 0.0043 |

Ranking: **XGBoost > SVM > Logistic Regression > Random Forest**.

Holdout (single 80/20 split, XGBoost): κ `0.9767`, macro-F1 `0.9534`, weighted
precision `0.9783`; macro one-vs-rest ROC-AUC `0.9992`, macro average precision
`0.9872`.

## 10. Statistical significance testing

25 paired folds (5 repeats × 5 folds), identical folds per comparison.
Differences are corrected with the Nadeau–Bengio resampled test
(`SE = √((1/J + n_test/n_train)·s²)`, `df = J − 1`), and the reported confidence
intervals use the **same** corrected variance, so intervals and p-values are
mutually consistent.

| Baseline | Metric | Mean Δ | 95% CI | Cohen's d | p (NB) | Significant |
|---|---|---:|---|---:|---:|---|
| SVM | Weighted F1 | +0.0060 | [0.0017, 0.0103] | 1.55 | 0.008 | yes |
| SVM | Accuracy | +0.0061 | [0.0020, 0.0102] | 1.64 | 0.006 | yes |
| SVM | Cohen's κ | +0.0064 | [0.0021, 0.0107] | 1.64 | 0.006 | yes |
| SVM | Macro-F1 | +0.0070 | [−0.0145, 0.0283] | 0.36 | 0.509 | **no** |
| Logistic Regression | Weighted F1 | +0.0077 | [0.0031, 0.0123] | 1.88 | 0.002 | yes |
| Logistic Regression | Accuracy | +0.0076 | [0.0031, 0.0120] | 1.89 | 0.002 | yes |
| Logistic Regression | Cohen's κ | +0.0080 | [0.0033, 0.0126] | 1.90 | 0.002 | yes |
| Logistic Regression | Macro-F1 | +0.0176 | [−0.0060, 0.0413] | 0.83 | 0.137 | **no** |

**Conclusions we do and do not draw.**

- ✅ XGBoost significantly outperforms **both** SVM and Logistic Regression on
  weighted F1, accuracy and κ.
- ❌ A statistically significant **macro-F1** advantage is **not** established.
  Macro-F1 fold variance is dominated by the two smallest classes.
- Plain Wilcoxon p-values are also recorded in
  `results/statistical_analysis/` but assume independent folds and are
  anti-conservative under repeated CV; only the Nadeau–Bengio values are
  reportable.

## 11. Per-class performance

Full per-class precision/recall/F1/support for all 32 classes:
[`results/metrics/per_class.csv`](results/metrics/per_class.csv)
(macro avg F1 `0.9461`, weighted avg F1 `0.9720`).

Classes below F1 = 0.90:

| Class | N (train) | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| OV — ovarian serous cystadenocarcinoma | 7 | 1.000 | **0.429** | 0.600 |
| READ — rectum adenocarcinoma | 73 | 0.824 | **0.575** | 0.677 |
| UCS — uterine carcinosarcoma | 44 | 0.907 | 0.886 | 0.897 |

Supports above refer to the **training partition** (6,651 samples). Class sizes
in the full 8,314-sample cohort are larger (e.g. OV: 9, READ: 91).

## 12. SHAP explainability

- TreeSHAP on a 200-sample subset of the held-out test set.
- Mean absolute SHAP value aggregated over samples and classes.
- Compared against XGBoost split-gain importance.

The two rankings disagree in a useful way: gain importance is dominated by mRNA
transcripts, while SHAP additionally surfaces miRNA markers (`hsa-miR-206`,
`hsa-miR-122-5p`). See
[`results/metrics/shap_top20.csv`](results/metrics/shap_top20.csv) and
[`results/metrics/feature_importance.csv`](results/metrics/feature_importance.csv).

## 13. Over-representation analysis (ORA)

Top-N mRNA genes by mean |SHAP| are tested against human cell-type signatures
from **CellMarker 2.0** using an exact hypergeometric test, with:

- the background restricted to the **3,217 measured mRNA genes** (not the genome),
- gene sets with zero background overlap excluded before testing,
- **Benjamini–Hochberg** FDR correction across all tested cell types.

Results are reported as −log₁₀(adjusted p). The enrichment is **cutoff
sensitive**: significance (FDR < 0.05) holds at N = 150 but not at N = 100 or
N = 200, so it is reported as an exploratory finding.

**These analyses are associational, not causal.** SHAP and ORA describe how
features contribute to model predictions and which annotated cell types their
targets overlap; they do not establish that any gene or pathway drives tumor
development.

## 14. Modality ablation (leave-one-modality-out)

XGBoost, identical splits and seeds; CV within the training partition.
Last two columns: paired difference versus the full model over 25 folds,
Nadeau–Bengio corrected.

| Configuration | #Features | CV acc | CV macro-F1 | Δ macro-F1 vs full | p |
|---|---:|---:|---:|---|---:|
| Full multi-omics | 9,844 | **0.9725** | 0.9443 | — | — |
| without mRNA | 6,627 | 0.9681 | 0.9454 | +0.0064 [−0.013, 0.026] | 0.507 |
| without miRNA | 9,461 | 0.9645 | 0.9259 | +0.0125 [−0.004, 0.029] | 0.128 |
| without CNV | 6,739 | 0.9722 | 0.9479 | −0.0016 [−0.019, 0.015] | 0.851 |
| without methylation | 6,705 | 0.9657 | 0.9296 | +0.0093 [−0.001, 0.020] | 0.078 |
| mRNA-only | 3,217 | 0.9547 | 0.9038 | **+0.0368 [0.025, 0.049]** | **<0.001** |

**What this shows — and what it does not.**

- ✅ An **mRNA-only** model is significantly worse than the full multi-omics
  model on both accuracy and macro-F1. Multi-omics integration helps.
- ❌ Removing **any single modality** does not significantly change macro-F1.
- ❌ mRNA is therefore **not** individually necessary, despite dominating the
  importance rankings; the discriminative signal is distributed across
  modalities.
- ❌ We do **not** claim that all four modalities are individually necessary, nor
  that removing mRNA improves performance (the point estimate is positive but
  the interval spans zero).

The defensible conclusion is the one the paper makes: **global feature
importance does not imply modality-level predictive necessity.**

## 15. Limitations

- **Fold dependence.** The 25 repeated-CV folds are not independent. Only
  Nadeau–Bengio corrected p-values are reportable.
- **Macro-F1 superiority not established.** XGBoost's advantage is significant
  on weighted metrics only.
- **Small classes.** OV (n = 7 in the training partition) and READ (n = 73) drive
  most of the macro-F1 variance.
- **Cutoff sensitivity.** ORA significance holds only at N = 150.
- **No external validation.** Public resources such as GEO and ICGC lack uniform
  coverage of all four modalities, which would confound validation with cohort
  composition.
- **Fixed hyperparameters.** No tuning was performed, so absolute performance is
  not an upper bound.
- **Reported literature comparison.** CMGL and MOGONET numbers are taken from
  their publications, not re-run under a controlled head-to-head protocol.
- **Causal claims.** None are made.

## 16. Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── XGBoost_PanCancer_GPU.ipynb     # executed end-to-end pipeline
├── src/
│   ├── common.py                       # data loading + model factory
│   ├── run_xgboost.py                  # holdout + 5-fold CV benchmark
│   ├── interpret_xgboost.py            # checkpoint, gain importance, SHAP, ROC/PR
│   ├── repeated_cv_wilcoxon.py         # 25 paired folds (original protocol)
│   ├── metrics_kappa.py                # Cohen's κ + macro-F1
│   ├── ablation_mrna.py                # full vs mRNA-only
│   ├── make_class_mapping.py           # class id → TCGA code → cancer name
│   ├── ora_from_shap.py                # over-representation analysis
│   ├── revision_experiments.py         # corrected CV, SVM comparison, ablations
│   └── check_env.py                    # environment check
├── configs/
│   ├── environment-cpu.yml
│   └── environment-gpu.yml
├── results/
│   ├── metrics/                        # CV, per-class, ROC/PR, importance
│   ├── statistical_analysis/           # paired folds, corrected significance, ablations
│   ├── figures/                        # publication figures
│   └── superseded/                     # pre-revision (full-cohort CV) outputs
├── paper/
│   ├── revision/                       # verified replacement tables/blocks
│   ├── figures/
│   ├── archive/                        # pre-revision PDF
│   └── references.bib
├── docs/
│   └── methodology.md
└── data/
    └── README.md
```

`results/superseded/` holds the earlier full-cohort-CV outputs. They are retained
for provenance and **must not** be quoted in the paper.

## 17. Installation and execution

```bash
git clone <repository-url>
cd ICBEA-2027-PanCancer-MultiOmics

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python src/check_env.py
```

Place the MLOmics matrices in the repository root (or set `DATA_DIR`), then:

```bash
# Corrected benchmark reported in the paper
python src/revision_experiments.py          # CV in train split, XGB vs SVM vs LR, ablations

# Ablation uncertainty (25 paired folds) — takes several hours
python src/revision_ablation_uncertainty.py

# Supporting analyses
python src/run_xgboost.py                   # holdout + CV (original full-cohort protocol)
python src/interpret_xgboost.py             # SHAP, gain importance, ROC/PR
python src/make_class_mapping.py

# ORA (requires the external CellMarker list — see data/README.md)
python src/ora_from_shap.py --top 100,150,200
```

Or run the notebook end-to-end:

```bash
jupyter lab   # open notebooks/XGBoost_PanCancer_GPU.ipynb, Run All
```

> **Runtime note.** The notebook's 25-fold repeated-CV retraining cell loads the
> saved per-fold scores and recomputes the corrected statistics, because the full
> retraining takes hours (dominated by lbfgs Logistic Regression on 9,844
> features). The full implementation is `src/revision_experiments.py`.

## 18. Paper information

- **Title:** Lineage-Associated Interpretable Pan-Cancer Tumor-Type
  Classification using Multi-Omics Data
- **Paper ID:** A2026
- **Venue:** ICBEA 2027, Xi'an, China, 26–29 March 2027
- **Status:** accepted with revisions; camera-ready in preparation

Paper assets (archived PDF, revision blocks, figures, bibliography) are in
[`paper/`](paper/README.md). A formal citation will be added once the proceedings
are published.

## License

Released under the MIT License. See [`LICENSE`](LICENSE). The MLOmics dataset and
the CellMarker library retain their own licenses and are not redistributed here.
