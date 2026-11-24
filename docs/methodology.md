# Methodology

End-to-end description of the pipeline implemented in this repository.

## 1. Data and preprocessing

Four modality matrices are loaded (`mRNA`, `miRNA`, `CNV`, `Methy`), each with
genes/probes as rows and samples as columns.

1. **Transpose** so that rows are samples.
2. **Suffix** every feature name with its modality (`_mRNA`, `_miRNA`, `_CNV`,
   `_Methy`) to preserve provenance and avoid column collisions.
3. **Inner join** the four matrices on TCGA sample identifiers → 8,314 samples.
4. **Label alignment** to the joined sample order; labels are integers 0–31
   (32 tumor types).
5. **Imputation** — column-mean imputation applied as a defensive step. The
   published MLOmics matrices contain **no missing values** (verified across all
   four modalities), so this operation changes no value.
6. **Split** — stratified 80/20 train/test (`random_state=42`).
7. **Scaling** — `StandardScaler` fit on training data only, refit inside every
   cross-validation fold.

### Leakage control

The 80/20 split is created once. Every subsequent cross-validation runs
**strictly inside the 80% training partition**; the 20% holdout is used only for
the final evaluation and never influences model selection. Because the matrices
contain no missing values, the imputation step cannot propagate information from
the holdout into the training folds.

## 2. Classifiers

Four classifiers are trained on the identical standardized feature matrix:

| Model | Configuration |
|---|---|
| XGBoost | `n_estimators=300` (200 in CV), `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, GPU `device='cuda'` with CPU fallback |
| Linear SVM | `SVC(kernel='linear')`, `random_state=42` |
| Random Forest | `n_estimators=300` (200 in CV), `random_state=42` |
| Logistic Regression | `max_iter=2000`, `solver='lbfgs'`, `random_state=42` |

Hyperparameters were deliberately **not** tuned, so that no model is advantaged
by the experimental design.

## 3. Evaluation

- **Primary:** 5-fold stratified cross-validation
  (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) within the
  training partition.
- **Metrics:** weighted precision, weighted F1, accuracy, Cohen's κ,
  macro-averaged F1.
- **Fold consistency:** `1 − σ/μ` of weighted precision across folds.
- **Confusion matrices** from `cross_val_predict` on the training partition.
- **Per-class report:** precision/recall/F1/support for all 32 classes.

Cohen's κ and macro-F1 are reported alongside weighted metrics because weighted
averages can hide failures on rare classes under the strong class imbalance
(9 to 757 samples per class, ≈84× skew).

## 4. Repeated CV and corrected statistical testing

A single 5-fold run gives only five observations per model — too few for a
meaningful paired test. The stratified 5-fold split is therefore repeated five
times with different shuffles (seeds 1000, 2000, 3000, 4000, 5000), producing
**25 paired folds**. XGBoost, the linear SVM and Logistic Regression are trained
on identical folds with identical per-fold scaling, so both comparisons
(XGBoost vs SVM, XGBoost vs LR) are paired.

### Why a correction is required

Repeated-CV folds are **not independent**: each fold's training set overlaps
substantially with the others. A Wilcoxon or paired t-test that assumes
independence understates the variance and inflates significance. We therefore
report the **Nadeau–Bengio corrected resampled test**:

```
SE_NB = sqrt( (1/J + n_test/n_train) · s_d² )
t     = mean_d / SE_NB
df    = J − 1
```

with `J = 25` repetitions and `n_test/n_train = 1/4` for 5-fold. The reported
95% confidence intervals use the **same** corrected standard error, so each
interval is consistent with its p-value. Effect size is reported as the paired
Cohen's d. The uncorrected Wilcoxon p-value is also recorded, for reference only.

## 5. Interpretability

- **Split-gain importance** — XGBoost `feature_importances_`.
- **SHAP** — `shap.TreeExplainer` on a 200-sample subset of the test set; mean
  absolute SHAP value aggregated over samples and classes.
- The two rankings are compared. Gain importance is dominated by mRNA
  transcripts; SHAP additionally surfaces miRNA markers. A high rank in either
  ranking is **not** evidence that a modality is independently sufficient — that
  question is answered by the ablation in §6.

## 6. Leave-one-modality-out ablation

The single-modality comparison (full vs mRNA-only) is extended to a full
leave-one-modality-out design. Six configurations are trained with identical
protocol and seeds:

| Configuration | Features |
|---|---:|
| Full multi-omics | 9,844 |
| without mRNA | 6,627 |
| without miRNA | 9,461 |
| without CNV | 6,739 |
| without methylation | 6,705 |
| mRNA-only | 3,217 |

Each configuration is evaluated on the holdout and with 5-fold CV inside the
training partition. To decide whether a difference is real rather than
fold noise, the six configurations are also compared **pairwise against the full
model over 25 repeated folds**, with the same Nadeau–Bengio correction.

The result is that only the mRNA-only model differs significantly from the full
model; removing any individual modality does not significantly change macro-F1.
This is the evidence behind the paper's central interpretability claim: global
feature importance does not imply modality-level predictive necessity.

## 7. Over-representation analysis (ORA)

Implemented in `src/ora_from_shap.py`.

1. The SHAP ranking is filtered to mRNA features (`_mRNA` suffix) and re-ranked
   by mean |SHAP| → `ORA_results/ranked_mrna_shap.csv`.
2. Top-N gene lists (default **N = 100, 150, 200**) are generated.
3. Enrichment of each list is tested against human cell-type marker sets from
   **CellMarker 2.0** using an exact **hypergeometric** test.
4. The background universe is restricted to the **3,217 mRNA genes actually
   measured** in the dataset — not the whole genome — so unmeasured targets
   cannot inflate significance.
5. Gene sets with zero background overlap are excluded before testing.
6. p-values are corrected with the **Benjamini–Hochberg** FDR across all tested
   cell types, and results are reported on the −log₁₀(adjusted p) scale.

The enrichment is cutoff sensitive: significance (FDR < 0.05) holds at N = 150
but not at N = 100 or N = 200, so it is reported as an exploratory finding.

**Interpretation.** SHAP and ORA provide model-level predictive explanations and
statistical associations with annotated cell types. They do not establish causal
relationships between genes, pathways and tumor development.

```bash
python src/ora_from_shap.py --top 100,150,200
```

> Requires `Pan-cancer_mRNA.csv` and the SHAP table in the working directory, and
> the CellMarker list at `human_cell_marker_/human_cell_marker.txt`
> (see `data/README.md`).
