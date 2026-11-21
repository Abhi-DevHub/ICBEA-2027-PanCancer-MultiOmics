r"""STEP 2 + 3 — Repeated 5-fold CV with paired XGBoost vs Logistic Regression
and the Wilcoxon signed-rank test (INDICON 2026 revision).

Generates 5 repeats x 5 folds = 25 paired observations per metric, satisfying
Ma'am's ">= 6 paired observations" requirement, then tests whether XGBoost is
significantly better than Logistic Regression on the SAME folds.

Usage:
    ..\venv\Scripts\python.exe repeated_cv_wilcoxon.py [--repeats 5] [--folds 5] [--data-dir DIR]

Outputs:
    repeated_cv_paired.csv   (25 rows x {repeat, fold, model metric columns})
    wilcoxon_results.csv     (per metric: n, W, p, mean delta, conclusion)
    paired_scores.png        (paired XGB vs LR F1 box/line plot)
"""
import argparse
import os

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import load_data, make_xgb

METRICS = ["precision_weighted", "f1_weighted", "accuracy", "kappa"]


def fold_metrics(y_true, y_pred):
    return {
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    print("Loading data...")
    X, y = load_data(args.data_dir)
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {y.nunique()}")

    rows = []
    for rep in range(args.repeats):
        seed = 1000 * (rep + 1)  # deterministic, distinct per repeat
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            scaler = StandardScaler().fit(X.iloc[tr])
            Xtr = scaler.transform(X.iloc[tr])
            Xte = scaler.transform(X.iloc[te])

            xgb_model = make_xgb(200)
            xgb_model.fit(Xtr, y.iloc[tr])
            xgb_pred = xgb_model.predict(Xte)

            lr_model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=1, solver="lbfgs")
            lr_model.fit(Xtr, y.iloc[tr])
            lr_pred = lr_model.predict(Xte)

            row = {"repeat": rep, "fold": fold}
            for m in METRICS:
                row[f"xgb_{m}"] = fold_metrics(y.iloc[te], xgb_pred)[m]
                row[f"lr_{m}"] = fold_metrics(y.iloc[te], lr_pred)[m]
            rows.append(row)
            print(f"repeat={rep} fold={fold}  XGB F1={row['xgb_f1_weighted']:.4f}  "
                  f"LR F1={row['lr_f1_weighted']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("repeated_cv_paired.csv", index=False)
    print(f"\nSaved repeated_cv_paired.csv ({len(df)} paired folds)")

    # --- Wilcoxon signed-rank on paired differences ---
    results = []
    for m in METRICS:
        xgb_scores = df[f"xgb_{m}"]
        lr_scores = df[f"lr_{m}"]
        diff = xgb_scores - lr_scores
        n = len(diff)
        # scipy >= 1.10 returns StatisticResult; handle both
        w_res = stats.wilcoxon(xgb_scores, lr_scores, alternative="greater")
        W = float(w_res.statistic)
        p = float(w_res.pvalue)
        mean_delta = float(diff.mean())
        conclusion = "XGBoost significantly better" if p < 0.05 else "No significant difference (p >= 0.05)"
        results.append({
            "metric": m,
            "n_paired": n,
            "W": round(W, 4),
            "p": round(p, 6),
            "mean_delta_xgb_minus_lr": round(mean_delta, 5),
            "xgb_mean": round(float(xgb_scores.mean()), 5),
            "lr_mean": round(float(lr_scores.mean()), 5),
            "conclusion": conclusion,
        })
        print(f"\n{m}: n={n} W={W:.3f} p={p:.6f} mean_delta={mean_delta:.5f} -> {conclusion}")

    res = pd.DataFrame(results)
    res.to_csv("wilcoxon_results.csv", index=False)
    print("\nSaved wilcoxon_results.csv")

    # --- Paired plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df[["xgb_f1_weighted", "lr_f1_weighted"]], ax=axes[0])
    axes[0].set_title("Weighted F1 over 25 paired folds")
    axes[0].set_ylabel("Weighted F1")
    axes[0].set_xticklabels(["XGBoost", "Logistic Regression"])
    sns.scatterplot(data=df, x="lr_f1_weighted", y="xgb_f1_weighted", ax=axes[1], s=40)
    lim = [df[["xgb_f1_weighted", "lr_f1_weighted"]].min().min() - 0.001,
           df[["xgb_f1_weighted", "lr_f1_weighted"]].max().max() + 0.001]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_title("Paired fold F1 (points above line = XGBoost better)")
    axes[1].set_xlabel("LR weighted F1")
    axes[1].set_ylabel("XGBoost weighted F1")
    plt.tight_layout()
    plt.savefig("paired_scores.png", dpi=300, bbox_inches="tight")
    print("Saved paired_scores.png")


if __name__ == "__main__":
    main()