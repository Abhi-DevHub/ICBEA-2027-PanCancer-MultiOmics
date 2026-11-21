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
5. **Imputation** — column-mean imputation for any residual missing value
   (the published matrices are largely pre-imputed).
6. **Split** — stratified 80/20 train/test (`random_state=42`).
7. **Scaling** — `StandardScaler` fit on training data only; refit inside every
   cross-validation fold so no test information leaks.

## 2. Classifiers

Four classifiers are trained on the identical standardized feature matrix:

| Model | Configuration |
|---|---|
| XGBoost | `n_estimators=300` (200 in CV), `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, GPU `device='cuda'` with CPU fallback |
| Linear SVM | `SVC(kernel='linear')`, `random_state=42` |
| Random Forest | `n_estimators=300` (200 in CV), `random_state=42` |
| Logistic Regression | `max_iter=2000`, `solver='lbfgs'`, `random_state=42` |

Hyperparameters were deliberately **not** tuned, so that no model is advantaged
by the experimental design; the comparison is a like-for-like baseline.

## 3. Evaluation

- **Primary:** 5-fold stratified cross-validation
  (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Metrics:** weighted precision, weighted F1, accuracy, Cohen's κ,
  macro-averaged F1.
- **Fold consistency:** `1 − σ/μ` of the metric across folds.
- **Confusion matrices** from `cross_val_predict`.
- **Per-class report:** precision/recall/F1/support for all 32 classes.

Cohen's κ and macro-F1 are reported alongside weighted metrics because weighted
averages can hide failures on rare classes under the strong class imbalance
(9 to 757 samples per class, ≈84× skew).

## 4. Paired statistical validation

A single 5-fold run gives only five observations per model — too few for a
meaningful paired test. The stratified 5-fold split is therefore repeated five
times with different shuffles (seeds 1000, 2000, 3000, 4000, 5000), producing
**25 paired folds**. XGBoost and Logistic Regression are trained on identical
folds with identical per-fold scaling.

A **one-sided Wilcoxon signed-rank test** (`alternative="greater"`) is applied to
the 25 paired differences in weighted precision, weighted F1, accuracy and
Cohen's κ.

> Caveat: the 25 folds are not strictly independent, because repeated-CV folds
> share training samples. The test is therefore interpreted as evidence that the
> advantage is consistent across partitions, with the mean differences used for
> effect size.

## 5. Single-modality ablation

To test whether the non-mRNA modalities contribute beyond mRNA expression, an
XGBoost model is trained on the **3,217 mRNA features only** and compared with
the full **9,844-feature** model under an identical protocol and seeds.
Holdout and CV accuracy and macro-F1 are reported for both.

## 6. Interpretability

- **Split-gain importance** — XGBoost `feature_importances_`.
- **SHAP** — `shap.TreeExplainer` on a 200-sample subset of the test set;
  mean absolute SHAP value aggregated over samples and classes.
- The two rankings are compared, and the ablations are used to check whether high
  importance implies that a modality is *sufficient* on its own.

## 7. Over-representation analysis (ORA)

Implemented in `src/ora_from_shap.py`.

1. The SHAP ranking from step 6 is filtered to mRNA features (`_mRNA` suffix) and
   re-ranked by mean |SHAP| → `ORA_results/ranked_mrna_shap.csv`.
2. Top-N gene lists (default **N = 100, 150, 200**) are written to
   `ORA_results/top_{N}_genes.txt`.
3. Enrichment of each list is tested against human cell-type marker sets from
   **CellMarker 2.0** using an exact **hypergeometric** test.
4. The background universe is restricted to the **3,217 mRNA genes actually
   measured** in the dataset — not the whole genome — so unmeasured targets
   cannot inflate significance.
5. p-values are corrected with the **Benjamini–Hochberg** FDR. Results are
   reported on the −log10(adjusted p) scale; the paper's 1.5 threshold on this
   scale corresponds to an FDR of ≈0.03.
6. Outputs in `ORA_results/`: `ora_top{N}_raw.csv`, `ora_top{N}_adj.csv`,
   `ora_top{N}_significant.csv`, `ora_top{N}_dotplot.png`, `ORA_metadata.json`.

```bash
python src/ora_from_shap.py --top 100,150,200
```

> Requires `Pan-cancer_mRNA.csv` and `shap_top20.csv` in the working directory,
> and the CellMarker list at `human_cell_marker_/human_cell_marker.txt`
> (see `data/README.md`).
