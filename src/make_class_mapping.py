r"""STEP 6 — Class 0-31 -> TCGA project code -> cancer name mapping (INDICON 2026).

The label file contains only integers; the repo has no name mapping. But every
sample id is a TCGA barcode (e.g. TCGA-IB-AAUW-01), so the project code is
recoverable data-driven. Output is a documentation artifact, not an ML experiment.

Usage:
    ..\venv\Scripts\python.exe make_class_mapping.py [--data-dir DIR]

Outputs:
    class_mapping_0_31.csv
"""
import argparse
import os

from common import load_data, build_class_mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    print("Loading data...")
    X, y = load_data(args.data_dir)
    mapping = build_class_mapping(X, y)
    mapping.to_csv("class_mapping_0_31.csv", index=False)
    print("Saved class_mapping_0_31.csv")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()