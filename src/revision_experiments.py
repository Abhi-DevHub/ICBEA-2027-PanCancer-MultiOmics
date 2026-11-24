# -*- coding: utf-8 -*-
r"""ICBEA 2027 reviewer-response experiments.

Fixes/handles:
  C1  Leakage: all cross-validation now runs INSIDE the 80% training split.
  C2  XGBoost vs SVM (closer baseline) on the same 25 paired folds,
      plus Nadeau-Bengio corrected test, 95% CI and Cohen's d.
  C3  Complete per-class metrics + leave-one-modality-out ablations.

Outputs (all prefixed `revision_`; originals untouched):
  revision_5fold_cv_trainonly.csv         5-fold CV inside train split, 4 models
  revision_per_class_trainonly.csv        per-class P/R/F1/support (XGBoost, pooled CV on train)
  revision_paired_xgb_svm.csv             25 paired folds, XGB vs SVM
  revision_significance_corrected.csv     Wilcoxon + Nadeau-Bengio + CI + effect size (XGB vs SVM/LR)
  revision_ablation_leave_one_out.csv     full / -mRNA / -miRNA / -CNV / -Methy / mRNA-only

Usage:  ..\venv\Scripts\python.exe revision_experiments.py
"""
import os, sys, time
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             cohen_kappa_score, classification_report)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import load_data, make_xgb

METRICS = ["precision_weighted", "f1_weighted", "accuracy", "kappa", "macro_f1"]


def m(y_true, y_pred):
    return {
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def models_full():
    """All four models (300 trees for XGB/RF, matching the holdout config)."""
    return {
        "XGBoost": make_xgb(300),
        "SVM": SVC(kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, solver="lbfgs"),
    }


def main():
    t_start = time.time()
    print("Loading data...")
    X, y = load_data(None)
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {y.nunique()}")

    # ---- the single 80/20 split used everywhere ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}  (CV will run ONLY on train)")

    # =====================================================================
    # C1 — 5-fold CV restricted to the 80% training split (all 4 models)
    # =====================================================================
    print("\n=== [C1] 5-fold CV inside training split (4 models) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, est in models_full().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
        per_fold = {k: [] for k in METRICS}
        t0 = time.time()
        for tr, te in cv.split(X_train, y_train):
            pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
            pred = pipe.predict(X_train.iloc[te])
            fm = m(y_train.iloc[te], pred)
            for k in METRICS:
                per_fold[k].append(fm[k])
        row = {"model": name}
        for k in METRICS:
            row[f"cv_mean_{k}"] = round(float(np.mean(per_fold[k])), 5)
            row[f"cv_std_{k}"] = round(float(np.std(per_fold[k], ddof=1)), 5)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  {name:<20} acc={row['cv_mean_accuracy']:.5f}  f1={row['cv_mean_f1_weighted']:.5f}  "
              f"macroF1={row['cv_mean_macro_f1']:.5f}  ({row['seconds']}s)")
    pd.DataFrame(rows).to_csv("revision_5fold_cv_trainonly.csv", index=False)
    print("  saved revision_5fold_cv_trainonly.csv")

    # per-class report (XGBoost, pooled CV predictions on TRAIN only)
    print("\n=== [C3] per-class metrics (XGBoost, pooled CV on train) ===")
    xgb_pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_xgb(200))])
    y_cv_pred = cross_val_predict(xgb_pipe, X_train, y_train, cv=cv, n_jobs=1)
    rep = classification_report(y_train, y_cv_pred, output_dict=True, zero_division=0)
    pc_rows = []
    for k, v in rep.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        pc_rows.append({"class": int(float(k)), "support": int(v["support"]),
                        "precision": round(v["precision"], 4),
                        "recall": round(v["recall"], 4),
                        "f1": round(v["f1-score"], 4)})
    pc = pd.DataFrame(pc_rows).sort_values("class")
    pc.to_csv("revision_per_class_trainonly.csv", index=False)
    print(f"  saved revision_per_class_trainonly.csv ({len(pc)} classes)")
    print(f"  macro avg F1={rep['macro avg']['f1-score']:.4f} | weighted avg F1={rep['weighted avg']['f1-score']:.4f}")

    # =====================================================================
    # C2 — 25 paired folds: XGBoost vs SVM (and LR) on IDENTICAL folds
    # =====================================================================
    print("\n=== [C2] 25 paired folds inside train: XGB vs SVM vs LR ===")
    paired = []
    for rep_i in range(5):
        seed = 1000 * (rep_i + 1)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(X_train, y_train)):
            sc = StandardScaler().fit(X_train.iloc[tr])
            Xtr, Xte = sc.transform(X_train.iloc[tr]), sc.transform(X_train.iloc[te])
            ytr, yte = y_train.iloc[tr], y_train.iloc[te]

            mx = make_xgb(200); mx.fit(Xtr, ytr)
            ms = SVC(kernel="linear", random_state=42); ms.fit(Xtr, ytr)
            ml = LogisticRegression(max_iter=2000, random_state=42, n_jobs=1, solver="lbfgs"); ml.fit(Xtr, ytr)

            fx, fs, fl = m(yte, mx.predict(Xte)), m(yte, ms.predict(Xte)), m(yte, ml.predict(Xte))
            row = {"repeat": rep_i, "fold": fold}
            for k in METRICS:
                row[f"xgb_{k}"] = fx[k]; row[f"svm_{k}"] = fs[k]; row[f"lr_{k}"] = fl[k]
            paired.append(row)
            print(f"  rep={rep_i} fold={fold}  XGB f1={fx['f1_weighted']:.4f}  "
                  f"SVM f1={fs['f1_weighted']:.4f}  LR f1={fl['f1_weighted']:.4f}  "
                  f"[{time.time()-t_start:.0f}s]")
    P = pd.DataFrame(paired)
    P.to_csv("revision_paired_xgb_svm.csv", index=False)
    print(f"  saved revision_paired_xgb_svm.csv ({len(P)} folds)")

    # =====================================================================
    # C2 — corrected significance: Wilcoxon + Nadeau-Bengio + CI + effect size
    # =====================================================================
    print("\n=== [C2] corrected statistics (Nadeau-Bengio resampled t) ===")
    J = len(P)                    # 25 resamples
    n_test_over_train = 1.0 / 4.0  # 5-fold -> test is 1/4 of train
    stat_rows = []
    for comp in ["svm", "lr"]:
        for k in METRICS:
            d = (P[f"xgb_{k}"] - P[f"{comp}_{k}"]).values
            n = len(d); mean_d = d.mean(); sd_d = d.std(ddof=1)
            # uncorrected (naive) paired t and wilcoxon
            t_naive = mean_d / (sd_d / np.sqrt(n)) if sd_d > 0 else np.nan
            p_naive = 2 * stats.t.sf(abs(t_naive), df=n - 1) if np.isfinite(t_naive) else np.nan
            wr = stats.wilcoxon(P[f"xgb_{k}"], P[f"{comp}_{k}"], alternative="greater")
            # Nadeau-Bengio corrected resampled t-test
            denom = np.sqrt((1.0 / J + n_test_over_train) * sd_d ** 2)
            t_nb = mean_d / denom if denom > 0 else np.nan
            p_nb = stats.t.sf(t_nb, df=n - 1) if np.isfinite(t_nb) else np.nan   # one-sided
            # 95% CI (t, df=n-1) and Cohen's d (paired)
            half = stats.t.ppf(0.975, n - 1) * sd_d / np.sqrt(n)
            cohen_d = mean_d / sd_d if sd_d > 0 else np.nan
            stat_rows.append({
                "comparison": f"XGBoost_vs_{comp.upper()}",
                "metric": k, "n": n,
                "xgb_mean": round(float(P[f'xgb_{k}'].mean()), 5),
                f"{comp}_mean": round(float(P[f'{comp}_{k}'].mean()), 5),
                "mean_delta": round(float(mean_d), 5),
                "sd_delta": round(float(sd_d), 5),
                "ci95_low": round(float(mean_d - half), 5),
                "ci95_high": round(float(mean_d + half), 5),
                "cohens_d": round(float(cohen_d), 3),
                "wilcoxon_W": round(float(wr.statistic), 1),
                "wilcoxon_p_one_sided": round(float(wr.pvalue), 6),
                "t_naive_p_two_sided": round(float(p_naive), 6),
                "t_nadeau_bengio_p_one_sided": round(float(p_nb), 6),
            })
    S = pd.DataFrame(stat_rows)
    S.to_csv("revision_significance_corrected.csv", index=False)
    print("  saved revision_significance_corrected.csv")
    print(S[S.metric == "f1_weighted"].to_string(index=False))

    # =====================================================================
    # C3 — leave-one-modality-out ablations
    # =====================================================================
    print("\n=== [C3] leave-one-modality-out ablations ===")
    suffixes = {"mRNA": "_mRNA", "miRNA": "_miRNA", "CNV": "_CNV", "Methy": "_Methy"}
    cols = {k: [c for c in X.columns if c.endswith(s)] for k, s in suffixes.items()}
    setups = [("full_multi_omics", list(X.columns))]
    for drop, c in cols.items():
        setups.append((f"without_{drop}", [x for x in X.columns if x not in set(c)]))
    setups.append(("mRNA_only", cols["mRNA"]))

    abl_rows = []
    for name, feature_list in setups:
        XX = X[feature_list]
        Xtr, Xte, ytr, yte = train_test_split(XX, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler().fit(Xtr)
        mx = make_xgb(300); mx.fit(sc.transform(Xtr), ytr)
        ho = m(yte, mx.predict(sc.transform(Xte)))
        # 5-fold CV inside train
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cvf = {k: [] for k in ["accuracy", "macro_f1", "f1_weighted"]}
        for tr, te in kf.split(Xtr, ytr):
            s2 = StandardScaler().fit(Xtr.iloc[tr])
            mm = make_xgb(200); mm.fit(s2.transform(Xtr.iloc[tr]), ytr.iloc[tr])
            fm = m(ytr.iloc[te], mm.predict(s2.transform(Xtr.iloc[te])))
            for k in cvf:
                cvf[k].append(fm[k])
        abl_rows.append({
            "configuration": name, "n_features": XX.shape[1],
            "holdout_accuracy": round(ho["accuracy"], 4),
            "holdout_macro_f1": round(ho["macro_f1"], 4),
            "cv_accuracy": round(float(np.mean(cvf["accuracy"])), 4),
            "cv_macro_f1": round(float(np.mean(cvf["macro_f1"])), 4),
            "cv_weighted_f1": round(float(np.mean(cvf["f1_weighted"])), 4),
        })
        print(f"  {name:<18} n={XX.shape[1]:>5}  hold acc={ho['accuracy']:.4f}  "
              f"CV acc={np.mean(cvf['accuracy']):.4f}  CV macroF1={np.mean(cvf['macro_f1']):.4f}")
    pd.DataFrame(abl_rows).to_csv("revision_ablation_leave_one_out.csv", index=False)
    print("  saved revision_ablation_leave_one_out.csv")

    print(f"\nALL DONE in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
