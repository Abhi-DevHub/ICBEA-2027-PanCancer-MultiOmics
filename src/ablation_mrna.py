r"""STEP 4 — Single-modality ablation: full multi-omics vs mRNA-only (INDICON 2026).

Tests whether mRNA expression alone (3,217 features) preserves the full-model
(9,844 features) classification performance — directly answering Ma'am's question
and the observation that all top-20 gain features are mRNA lineage markers.

Usage:
    ..\venv\Scripts\python.exe ablation_mrna.py [--data-dir DIR]

Outputs:
    ablation_comparison.csv   (per model-set: holdout + CV metrics)
    ablation_barplot.png      (grouped bar chart accuracy / macro-F1 / weighted-F1)
"""
import argparse
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from common import load_data, make_xgb


def eval_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def run_holdout(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    m = make_xgb(300)
    m.fit(X_tr_s, y_tr)
    pred = m.predict(X_te_s)
    return eval_metrics(y_te, pred)


def run_cv(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cols = {k: [] for k in ["accuracy", "macro_f1", "weighted_f1", "kappa"]}
    for tr, te in cv.split(X, y):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        m = make_xgb(200)
        m.fit(Xtr, y.iloc[tr])
        pred = m.predict(Xte)
        for k in cols:
            cols[k].append(eval_metrics(y.iloc[te], pred)[k])
    return {k: float(np.mean(v)) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    print("Loading data...")
    X, y = load_data(args.data_dir)

    mrna_cols = [c for c in X.columns if c.endswith("_mRNA")]
    X_mrna = X[mrna_cols]
    print(f"Full: {X.shape[1]} features | mRNA-only: {X_mrna.shape[1]} features")

    rows = []
    for name, XX in [("full_multi_omics", X), ("mRNA_only", X_mrna)]:
        print(f"\n=== {name} (holdout + 5-fold CV) ===")
        ho = run_holdout(XX, y)
        cv = run_cv(XX, y)
        row = {"setup": name, "n_features": XX.shape[1]}
        for k in sorted(ho):
            row[f"holdout_{k}"] = round(ho[k], 4)
        for k in sorted(cv):
            row[f"cv_mean_{k}"] = round(cv[k], 4)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv("ablation_comparison.csv", index=False)
    print("\nSaved ablation_comparison.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = df["setup"].tolist()
    x = np.arange(len(sets))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, metric in enumerate(["holdout_accuracy", "holdout_macro_f1", "holdout_weighted_f1"]):
        vals = df[metric].astype(float).values
        ax.bar(x + (i - 1) * width, vals, width, label=metric.replace("holdout_", ""))
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full\nmulti-omics", "mRNA only"], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("XGBoost: Full multi-omics vs mRNA-only (holdout, n=1663)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ablation_barplot.png", dpi=300, bbox_inches="tight")
    print("Saved ablation_barplot.png")


if __name__ == "__main__":
    main()