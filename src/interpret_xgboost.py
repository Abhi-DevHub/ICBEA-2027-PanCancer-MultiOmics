r"""
XGBoost interpretability + checkpoint module (XGBoost_Run, self-contained).

Re-trains the exact XGBoost model from run_xgboost.py (same data pipeline,
same hyperparameters, seed 42), then:

1. Saves the trained model checkpoint   -> xgboost_model.json
2. Feature importance (gain)            -> feature_importance.csv + feature_importance.png
3. SHAP analysis (TreeExplainer)        -> shap_summary.png + shap_beeswarm.png + shap_top20.csv
4. Multiclass ROC-AUC (macro, OVR)      -> roc_auc.png + roc_auc.csv
5. Precision-Recall (macro)             -> pr_curve.png + pr_curve.csv

Uses GPU (device='cuda') with CPU fallback. Outputs are written to the
folder this script lives in.

Usage:
    ..\venv\Scripts\python.exe interpret_xgboost.py [--data-dir PATH]
"""
import argparse
import os
import warnings
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import shap


def resolve_data_dir(args_data_dir):
    local = os.path.dirname(os.path.abspath(__file__))
    if args_data_dir:
        return args_data_dir
    if os.path.exists(os.path.join(local, "Pan-cancer_mRNA.csv")):
        return local
    return r"D:\Test_XGboost"


def load_data(data_dir):
    cancer = "Pan-cancer"
    fm = {
        m: pd.read_csv(os.path.join(data_dir, f"{cancer}_{m}.csv"), index_col=0)
        for m in ["mRNA", "miRNA", "CNV", "Methy"]
    }
    labels_df = pd.read_csv(os.path.join(data_dir, f"{cancer}_label_num.csv"), header=0)
    labels = labels_df["Label"] if "Label" in labels_df.columns else labels_df.iloc[:, 0]

    comb = fm["mRNA"].T.add_suffix("_mRNA")
    for m in ["miRNA", "CNV", "Methy"]:
        comb = comb.join(fm[m].T.add_suffix(f"_{m}"), how="inner")
    if not labels.index.equals(comb.index):
        if len(labels) == len(comb):
            labels = pd.Series(labels.values, index=comb.index, name="Label")
        else:
            common = labels.index.intersection(comb.index)
            comb = comb.loc[common]
            labels = labels.loc[common]
    comb = comb.apply(pd.to_numeric, errors="coerce").fillna(comb.mean())
    return comb, labels


def make_xgb(n_estimators):
    probe = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist",
                              device="cuda", random_state=42)
    try:
        probe.fit(np.random.rand(16, 4), np.random.randint(0, 2, 16))
        print("XGBoost: CUDA device available.")
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            device="cuda", random_state=42, n_jobs=1,
        )
    except Exception:
        print("CUDA not available; using CPU.")
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=42, n_jobs=1,
        )


def plot_top_features(importance, out_png, top_n=20):
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} XGBoost Features (gain importance)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_png)


def plot_multiclass_roc(model, X_test, y_test, n_classes, out_png, out_csv, sample=None):
    y_true = y_test.values
    probs = model.predict_proba(X_test)
    if sample and X_test.shape[0] > sample:
        idx = np.random.RandomState(42).choice(len(y_true), sample, replace=False)
        y_true = y_true[idx]
        probs = probs[idx]

    rows = []
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        if (y_true == i).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
        a = auc(fpr, tpr)
        rows.append({"class": i, "auc": a, "n": int((y_true == i).sum())})
        ax.plot(fpr, tpr, lw=1, alpha=0.6, label=f"class {i} (AUC {a:.3f})")

    macro = float(np.mean([r["auc"] for r in rows]))
    rows.append({"class": "macro", "auc": macro, "n": int(len(y_true))})
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Multiclass ROC (one-vs-rest) — macro AUC {macro:.3f}")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_png} (macro AUC {macro:.3f})")
    print(f"Saved {out_csv}")


def plot_multiclass_pr(model, X_test, y_test, n_classes, out_png, out_csv, sample=None):
    y_true = y_test.values
    probs = model.predict_proba(X_test)
    if sample and X_test.shape[0] > sample:
        idx = np.random.RandomState(42).choice(len(y_true), sample, replace=False)
        y_true = y_true[idx]
        probs = probs[idx]

    rows = []
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        if (y_true == i).sum() == 0:
            continue
        p, r, _ = precision_recall_curve((y_true == i).astype(int), probs[:, i])
        ap = average_precision_score((y_true == i).astype(int), probs[:, i])
        rows.append({"class": i, "ap": ap, "n": int((y_true == i).sum())})
        ax.plot(r, p, lw=1, alpha=0.6, label=f"class {i} (AP {ap:.3f})")

    macro = float(np.mean([r["ap"] for r in rows]))
    rows.append({"class": "macro", "ap": macro, "n": int(len(y_true))})
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Multiclass Precision-Recall (one-vs-rest) — macro AP {macro:.3f}")
    ax.legend(loc="lower left", fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_png} (macro AP {macro:.3f})")
    print(f"Saved {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--shap-sample", type=int, default=200,
                    help="Number of test samples to explain with SHAP (200 is fast).")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print("Data dir:", data_dir)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    X, y = load_data(data_dir)
    print(f"Data: {X.shape[0]} samples x {X.shape[1]} features, {y.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"Train: {X_train_s.shape[0]} | Test: {X_test_s.shape[0]}")

    t0 = time.time()
    if os.path.exists("xgboost_model.json"):
        model = xgb.XGBClassifier()
        model.load_model("xgboost_model.json")
        print("Loaded existing xgboost_model.json (skipping re-train).")
    else:
        model = make_xgb(300)
        model.fit(X_train_s, y_train)
        print(f"Trained XGBoost in {time.time() - t0:.1f}s")

    # 1. Save checkpoint
    model.save_model("xgboost_model.json")
    print("Saved xgboost_model.json")

    # 2. Feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv("feature_importance.csv", header=["importance"])
    plot_top_features(imp, "feature_importance.png")
    print("Top 5:", imp.head(5).index.tolist())

    # 3. SHAP
    print("Computing SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    sample_idx = np.random.RandomState(42).choice(X_test_s.shape[0],
                                                  min(args.shap_sample, X_test_s.shape[0]),
                                                  replace=False)
    X_shap = X_test_s[sample_idx]
    shap_values = explainer.shap_values(X_shap)
    print("SHAP values shape:", np.asarray(shap_values).shape)

    # Normalize to a list of per-class arrays (samples, features).
    # shap 0.52 multiclass may return (samples, features, classes).
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        per_class = [sv[:, :, i] for i in range(sv.shape[2])]
        mean_abs = np.mean(np.abs(sv), axis=(0, 2))
    else:
        per_class = sv
        mean_abs = np.mean(np.abs(sv), axis=0)

    top_shap = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    top_shap.to_csv("shap_top20.csv", header=["mean_abs_shap"])

    # summary + beeswarm using the highest-mean-impact class if multiclass
    disp = per_class[0] if isinstance(per_class, list) else per_class
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(disp, X_shap, feature_names=X.columns.tolist(),
                          show=False, max_display=20)
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved shap_summary.png")
    except Exception as e:
        print("summary plot skipped:", str(e)[:100])

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap.Explanation(values=disp, data=X_shap,
                                             feature_names=X.columns.tolist()),
                            max_display=20, show=False)
        plt.tight_layout()
        plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved shap_beeswarm.png")
    except Exception as e:
        print("beeswarm skipped:", str(e)[:100])

    # 4. ROC + PR
    plot_multiclass_roc(model, X_test_s, y_test, y.nunique(),
                        "roc_auc.png", "roc_auc.csv")
    plot_multiclass_pr(model, X_test_s, y_test, y.nunique(),
                       "pr_curve.png", "pr_curve.csv")

    print("\nAll interpretability outputs written to this folder.")


if __name__ == "__main__":
    main()