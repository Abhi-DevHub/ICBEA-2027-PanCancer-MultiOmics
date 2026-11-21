# Dataset

Raw data is **not** distributed with this repository.

## Source

**MLOmics: Cancer Multi-Omics Database for Machine Learning**
Yang, Z., Kotoge, R., Piao, X., et al. *Scientific Data* 12, 913 (2025).

- Paper: https://doi.org/10.1038/s41597-025-05235-x
- Database / code: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark
- Hugging Face mirror: https://huggingface.co/datasets/AIBIC/MLOmics

The benchmark is derived from The Cancer Genome Atlas (TCGA). Please cite the
MLOmics paper, and follow its license and the TCGA data-use terms when you
download the data.

## Files required by this code

Place these in the repository root, or set the `DATA_DIR` environment variable
to the folder that contains them:

| File | Description | Features |
|---|---|---|
| `Pan-cancer_mRNA.csv`   | mRNA expression   | 3,217 |
| `Pan-cancer_miRNA.csv`  | microRNA          | 383   |
| `Pan-cancer_CNV.csv`    | copy-number variation | 3,105 |
| `Pan-cancer_Methy.csv`  | DNA methylation   | 3,139 |
| `Pan-cancer_label_num.csv` | class labels (integers 0–31) | — |

After the inner join on TCGA sample identifiers the cohort is
**8,314 samples × 9,844 features**, covering **32 tumor types**.

## How paths are resolved

- Notebook: uses its own folder first, then `DATA_DIR`, then a local fallback.
- Scripts: accept `--data-dir`, then `DATA_DIR`, then a local fallback.

No credentials or tokens are required, and none are stored in this repository.
