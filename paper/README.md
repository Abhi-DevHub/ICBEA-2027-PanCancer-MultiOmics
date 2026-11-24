# Paper assets — ICBEA 2027

**Title:** Lineage-Associated Interpretable Pan-Cancer Tumor-Type Classification using Multi-Omics Data
**Paper ID:** A2026 · **Venue:** ICBEA 2027 (Xi'an, China, 26–29 March 2027)

## Status

| Item | State |
|---|---|
| `archive/pre_revision_paper.pdf` | Accepted version **before** the reviewer revisions (6 pages, contains the superseded CV numbers) |
| `revision/ICBEA_revision_tables.tex` | The 11 verified replacement blocks (tables + text) for the reviewer response |
| `references.bib` | Bibliography used by the paper |
| `figures/` | Publication figures (see note on the ORA figure below) |
| `main.tex` | **Not yet included** — the merged camera-ready source will be added once the revision blocks are integrated in Overleaf |
| `final_paper.pdf` | **Not yet included** — requires a LaTeX build of the merged source |

## How to produce the revised camera-ready

1. Open the Overleaf project.
2. Add `\usepackage{multirow}` to the preamble (Table IV uses `\multirow`).
3. Apply the blocks in `revision/ICBEA_revision_tables.tex` in order:
   - `[1]` replace Table II (`tab:cv`)
   - `[2]` replace Table III (`tab:kappa`)
   - `[3]` insert the new per-class table (`tab:perclass`)
   - `[4]` replace Table IV (`tab:wilcoxon`) — corrected, CI and *p* now consistent
   - `[5]` replace Table V (`tab:ablation`) — leave-one-modality-out with paired uncertainty
   - `[6]` insert the leakage/preprocessing paragraph in §III-A
   - `[7]` insert the non-causal interpretation paragraph in §III-E
   - `[8]` replace the ORA result sentence in §IV-E
   - `[9]` replace the abstract sentences
   - `[10]` replace the conclusion
   - `[11]` insert the statistical qualification into Limitations
4. Compile and confirm the 5-page limit.
5. Copy the resulting `main.tex` and `final_paper.pdf` into this directory.

## Numbers you must not reintroduce

The following values belong to the **superseded** full-cohort CV protocol and must
not appear in the revised paper or repository:

- CV accuracy `0.9756`, CV macro-F1 `0.9518`, CV κ `0.9744`
- Wilcoxon `W = 325` with `p < 0.001` presented as the primary test
- Ablation deltas `−1.65` / `−3.01` computed on the full-cohort CV

The corrected values and their statistical qualification are in
`results/metrics/` and `results/statistical_analysis/`.

## Note on the ORA figure

`figures/ora_top150_dotplot.png` is **not** included. The ORA statistics
(hypergeometric + Benjamini–Hochberg) are implemented and verified in
`src/ora_from_shap.py`, but regenerating the figure requires the external
CellMarker 2.0 gene list, which is not redistributed here (see `../data/README.md`).
