r"""STEP 5 — Add Cohen's kappa (+ macro-F1) to the 4-model benchmark (INDICON 2026).

Recomputes holdout + 5-fold CV for all four models with kappa and macro-F1, so the
paper table can include Cohen's kappa alongside precision/F1/accuracy. Uses the
same protocol and seeds as run_xgboost.py.

Usage:
    ..\venv\Scripts\python.exe metrics_kappa.py [--data-dir DIR]

Outputs:
    kappa_macro_results.csv   (holdout + CV rows for all 4 models)
"""
import argparse
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import load_data, make_xgb


def metrics(y_true, y_pred):
    return {
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    print("Loading data...")
    X, y = load_data(args.data_dir)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    model_builders = {
        "XGBoost": lambda: make_xgb(300),
        "SVM": lambda: SVC(kernel="linear", random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, solver="lbfgs"),
    }

    rows = []
    for name, build in model_builders.items():
        # Holdout
        m = build()
        m.fit(X_tr_s, y_tr)
        ho = metrics(y_te, m.predict(X_te_s))

        # 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_accum = {k: [] for k in metrics(np.array([0]), np.array([0]))}
        for tr, te in cv.split(X, y):
            s = StandardScaler().fit(X.iloc[tr])
            Xtr, Xte = s.transform(X.iloc[tr]), s.transform(X.iloc[te])
            mc = make_xgb(200) if name == "XGBoost" else build()
            mc.fit(Xtr, y.iloc[tr])
            for k, v in metrics(y.iloc[te], mc.predict(Xte)).items():
                cv_accum[k].append(v)

        row = {"model": name}
        for k in ho:
            row[f"holdout_{k}"] = round(ho[k], 4)
        for k in cv_accum:
            row[f"cv_mean_{k}"] = round(float(np.mean(cv_accum[k])), 4)
            row[f"cv_std_{k}"] = round(float(np.std(cv_accum[k])), 4)
        rows.append(row)
        print(f"{name:<20} holdout kappa={ho['kappa']:.4f} macroF1={ho['f1_macro']:.4f} "
              f"| CV kappa={row['cv_mean_kappa']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("kappa_macro_results.csv", index=False)
    print("\nSaved kappa_macro_results.csv")


if __name__ == "__main__":
    main()