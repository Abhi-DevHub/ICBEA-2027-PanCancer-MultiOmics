# Lineage-Associated Interpretable Pan-Cancer Tumor-Type Classification using Multi-Omics Data

Reference implementation for the ICBEA 2027 paper of the same title.

This repository contains the code, configurations and selected results needed to
reproduce the main computational workflow: 32-class pan-cancer tumor-type
classification on the **MLOmics** benchmark with four classical classifiers
(XGBoost, Linear SVM, Random Forest, Logistic Regression), paired statistical
validation, a single-modality ablation, and SHAP-based interpretability.

---

## Overview

Pan-cancer tumor-type classification from multi-omics data is a widely studied
problem, but reproducible and statistically validated benchmarks of classical
learners on structured omics tables remain scarce, and interpretability is often
reported without checking whether a modality is independently sufficient.

This work provides:

- a reproducible four-classifier benchmark on 8,314 samples × 9,844 features;
- chance-corrected (Cohen's κ) and class-balanced (macro-F1) reporting;
- a paired Wilcoxon signed-rank test over 25 repeated-CV folds;
- a single-modality (mRNA-only) ablation;
- SHAP-based feature importance with a comparison against split-gain importance.

## Methodology

The pipeline (`notebooks/XGBoost_PanCancer_GPU.ipynb` and `src/`) is:

1. **Preprocessing** — transpose each modality, suffix feature names by modality,
   inner-join the four matrices on TCGA sample identifiers, column-mean imputation,
   stratified 80/20 split, `StandardScaler` fit on training data only
   (refit inside each CV fold).
2. **Classification** — XGBoost, linear SVM, Random Forest, Logistic Regression on
   the identical standardized feature matrix; hyperparameters held fixed.
3. **Evaluation** — 5-fold stratified cross-validation; weighted precision/F1,
   accuracy, Cohen's κ, macro-F1; fold consistency `1 − σ/μ`.
4. **Paired statistical validation** — 5 repeats × 5 folds = 25 paired folds,
   one-sided Wilcoxon signed-rank test (XGBoost vs Logistic Regression).
5. **Single-modality ablation** — full 9,844 features vs mRNA-only 3,217.
6. **Interpretability** — SHAP (TreeExplainer) mean |SHAP| vs split-gain importance.

See `docs/methodology.md` for the full description.

## Dataset

The **MLOmics: Cancer Multi-Omics Database for Machine Learning** benchmark
(Yang et al., *Scientific Data* 12:913, 2025; derived from TCGA).

- Paper: https://doi.org/10.1038/s41597-025-05235-x
- Code/database: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark
- Parameters used here: **8,314 samples**, **32 tumor types**, four modalities —
  mRNA (3,217), miRNA (383), CNV (3,105), DNA methylation (3,139) = **9,844 features**.

> **Raw data is not distributed in this repository.** Download the MLOmics
> `Pan-cancer` matrices from the source above and place them in the repository
> root (or set the `DATA_DIR` environment variable to their location).

**External gene-set library (for ORA).** `src/ora_from_shap.py` additionally
needs a human cell-marker list from **CellMarker 2.0** (Hu et al., *Nucleic Acids
Research* 51:D870–D876, 2023), saved as
`human_cell_marker_/human_cell_marker.txt` (tab-delimited, with `cell_name` and
`symbol` columns). This file is not redistributed here — see `data/README.md`.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── XGBoost_PanCancer_GPU.ipynb   # executed end-to-end pipeline + results
├── src/
│   ├── common.py                     # shared data loading / model factory
│   ├── run_xgboost.py                # 4-model holdout + 5-fold CV benchmark
│   ├── interpret_xgboost.py          # checkpoint, gain importance, SHAP, ROC/PR
│   ├── repeated_cv_wilcoxon.py       # 25 paired folds + Wilcoxon test
│   ├── metrics_kappa.py              # Cohen's κ + macro-F1 (4 models)
│   ├── ablation_mrna.py              # mRNA-only vs full multi-omics
│   ├── make_class_mapping.py         # class 0–31 → TCGA code → cancer name
│   ├── ora_from_shap.py              # ORA: top-N SHAP mRNA genes vs CellMarker
│   └── check_env.py                  # environment / dependency check
├── configs/
│   ├── environment-cpu.yml
│   └── environment-gpu.yml
├── results/
│   ├── figures/                      # figures reported in the paper
│   └── *.csv                         # metric tables produced by the pipeline
├── docs/
│   └── methodology.md
└── data/
    └── README.md
```

## Installation

Requires **Python 3.10+** (developed on 3.13.12).

```bash
git clone <repository-url>
cd ICBEA-2027-PanCancer-MultiOmics

python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

python src/check_env.py     # verifies packages and dataset presence
```

## Execution

All commands are run **from the repository root** (`src/` scripts import
`common.py`, so run them with the repo root on the path):

```bash
# 1) Full benchmark: 4-model holdout + 5-fold CV + confusion matrices
python src/run_xgboost.py

# 2) Interpretability: model checkpoint, gain importance, SHAP, ROC/PR
python src/interpret_xgboost.py

# 3) Statistical validation: 25 paired folds + Wilcoxon signed-rank test
python src/repeated_cv_wilcoxon.py

# 4) Cohen's kappa + macro-F1 for all four models
python src/metrics_kappa.py

# 5) Single-modality ablation: mRNA-only vs full multi-omics
python src/ablation_mrna.py

# 6) Class 0–31 → TCGA project → cancer name
python src/make_class_mapping.py

# 7) Over-representation analysis (see note below on required inputs)
python src/ora_from_shap.py --top 100,150,200
```

> **ORA inputs.** `src/ora_from_shap.py` expects `Pan-cancer_mRNA.csv` and
> `shap_top20.csv` in the **current working directory**, plus the human
> CellMarker gene list at `human_cell_marker_/human_cell_marker.txt`
> (tab-delimited, with `cell_name` and `symbol` columns). The CellMarker file
> must be downloaded separately — see `data/README.md`. Outputs (raw/adjusted/
> significant tables, `ranked_mrna_shap.csv`, top-N gene lists and dot plots)
> are written to `ORA_results/`.

Or run the notebook end-to-end:

```bash
jupyter lab
# open notebooks/XGBoost_PanCancer_GPU.ipynb and run all cells
```

> **Runtime note.** In `notebooks/XGBoost_PanCancer_GPU.ipynb`, the 25-fold
> repeated-CV retraining cell is replaced by loading the saved per-fold scores
> (`results/repeated_cv_paired.csv`) and running the identical Wilcoxon test on
> them. The full retraining (hours, dominated by lbfgs LogisticRegression on
> 9,844 features) is implemented in `src/repeated_cv_wilcoxon.py`. All other
> notebook cells execute normally and produce the figures in `results/figures/`.

## Results

Reported values correspond to the CSVs in `results/`.

**5-fold stratified CV (weighted):**

| Model | Precision | F1 | Accuracy |
|---|---|---|---|
| **XGBoost** | **0.9758** | **0.9749** | **0.9756** |
| SVM | 0.9680 | 0.9671 | 0.9675 |
| Logistic Regression | 0.9655 | 0.9649 | 0.9656 |
| Random Forest | 0.9553 | 0.9495 | 0.9535 |

**Cohen's κ / macro-F1 (XGBoost):** holdout κ 0.9767, CV κ 0.9744 ± 0.0023;
holdout macro-F1 0.9534, CV macro-F1 0.9518 ± 0.0107.

**Paired Wilcoxon (25 folds, XGBoost vs Logistic Regression):** W = 325
(maximum possible), p < 0.001 on every metric; mean Δ precision +0.0113,
F1 +0.0109, accuracy +0.0105, κ +0.0110.

**Single-modality ablation:** CV accuracy 0.9756 → 0.9591 (−1.65 points);
CV macro-F1 0.9518 → 0.9217 (−3.01 points).

**Multiclass ROC/PR (holdout, n = 1,663):** macro one-vs-rest ROC-AUC 0.9992,
macro average precision 0.9872.

## Reproducibility

- **Random seeds:** `random_state = 42` for splits, models and scaling.
  Repeated CV uses seeds `1000, 2000, 3000, 4000, 5000`.
- **Software:** Python 3.13.12, numpy 2.3.2, pandas 2.3.1, scikit-learn 1.9.0,
  xgboost 3.4.1, scipy 1.18.0, matplotlib 3.11.1, seaborn 0.13.2, shap 0.52.0.
- **Hardware used:** NVIDIA RTX 4060 (8 GB), AMD Ryzen AI 9 HX 370 CPU.
  XGBoost runs on GPU via `device='cuda'`, with automatic CPU fallback.
- **Model configuration:** XGBoost `n_estimators=300` (200 in CV), `max_depth=6`,
  `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`;
  Random Forest `n_estimators=300` (200 in CV); linear SVM `kernel='linear'`;
  Logistic Regression `max_iter=2000, solver='lbfgs'`.
- **Note on class imbalance:** the rarest classes have very few samples
  (OV: n = 9; READ: n = 91), so macro-averaged metrics for those classes are
  sensitive to one or two observations.

## Scope

This release covers preprocessing, the four-classifier benchmark, statistical
validation, the single-modality ablation, class mapping, SHAP interpretability,
and the over-representation analysis (ORA) of the top SHAP-ranked mRNA genes
against curated cell-type markers.

The ORA implementation (`src/ora_from_shap.py`) requires an external human
CellMarker gene list that is not redistributed here (see `data/README.md`); its
outputs are reported in the paper.

## Citation

```bibtex
@inproceedings{icbea2027pancancer,
  title     = {Lineage-Associated Interpretable Pan-Cancer Tumor-Type
               Classification using Multi-Omics Data},
  author    = {Anil, Akhilesh and others},
  booktitle = {International Conference on Biomedical Engineering and Applications (ICBEA)},
  year      = {2027}
}
```

> Replace with the final published citation when available.

## License

Released under the MIT License. See `LICENSE`.
