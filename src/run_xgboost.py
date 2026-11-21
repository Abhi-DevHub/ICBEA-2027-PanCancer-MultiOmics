r"""
Runnable reproduction of the XGBoost/SVM/RF/LR pan-cancer experiment
(32-class classification on multi-omics, 5-fold stratified CV).

Faithful port of `Final/XGBoost_ SVM_RF_LR.ipynb` (cells 0-3), adapted for
xgboost 3.x GPU API (`device='cuda'`, `tree_method='hist'`) with automatic
fallback to CPU if CUDA is unavailable.

Usage (from the XGBoost_Run folder, using the GPU venv):
    ..\venv\Scripts\python.exe run_xgboost.py [--data-dir "D:\\Test_XGboost"]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb


def make_xgb(n_estimators):
    """XGBoost classifier using GPU when available, CPU fallback otherwise."""
    probe = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=1,
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    try:
        probe.fit(np.random.rand(16, 4), np.random.randint(0, 2, 16))
        print("XGBoost: CUDA device available.")
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=42,
            n_jobs=1,
        )
    except Exception:
        print("CUDA not available for XGBoost; falling back to CPU.")
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Folder containing Pan-cancer_*.csv. Defaults to this script's folder, "
        "falling back to D:\\Test_XGboost.",
    )
    args = ap.parse_args()

    local = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = local if os.path.exists(os.path.join(local, "Pan-cancer_mRNA.csv")) else r"D:\Test_XGboost"

    cancer = "Pan-cancer"
    file_map = {
        "mRNA": os.path.join(data_dir, f"{cancer}_mRNA.csv"),
        "miRNA": os.path.join(data_dir, f"{cancer}_miRNA.csv"),
        "CNV": os.path.join(data_dir, f"{cancer}_CNV.csv"),
        "Methy": os.path.join(data_dir, f"{cancer}_Methy.csv"),
        "label": os.path.join(data_dir, f"{cancer}_label_num.csv"),
    }
    for k, v in file_map.items():
        if not os.path.exists(v):
            raise FileNotFoundError(v)

    print("Loading data...")
    omics_file_1 = pd.read_csv(file_map["mRNA"], index_col=0)
    omics_file_2 = pd.read_csv(file_map["miRNA"], index_col=0)
    omics_file_3 = pd.read_csv(file_map["CNV"], index_col=0)
    omics_file_4 = pd.read_csv(file_map["Methy"], index_col=0)
    labels_df = pd.read_csv(file_map["label"], header=0)

    omics_data_1 = omics_file_1.T.add_suffix("_mRNA")
    omics_data_2 = omics_file_2.T.add_suffix("_miRNA")
    omics_data_3 = omics_file_3.T.add_suffix("_CNV")
    omics_data_4 = omics_file_4.T.add_suffix("_Methy")

    omics_data_combined = (
        omics_data_1.join(omics_data_2, how="inner")
        .join(omics_data_3, how="inner")
        .join(omics_data_4, how="inner")
    )
    print(f"Samples after inner-join: {omics_data_combined.shape[0]} | Features: {omics_data_combined.shape[1]}")

    if "Label" in labels_df.columns:
        labels = labels_df["Label"]
    else:
        labels = labels_df.iloc[:, 0]
    if not labels.index.equals(omics_data_combined.index):
        if len(labels) == len(omics_data_combined):
            labels = pd.Series(labels.values, index=omics_data_combined.index, name="Label")
        else:
            common_idx = labels.index.intersection(omics_data_combined.index)
            omics_data_combined = omics_data_combined.loc[common_idx]
            labels = labels.loc[common_idx]
    print(f"Samples with labels after alignment: {len(labels)}")

    omics_data_combined = omics_data_combined.apply(pd.to_numeric, errors="coerce")
    omics_data_combined = omics_data_combined.fillna(omics_data_combined.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        omics_data_combined, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_ordered = {
        "XGBoost": make_xgb(300),
        "SVM": SVC(kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, solver="lbfgs"),
    }

    def evaluate_model(model, Xtr, Xte, ytr, yte):
        start = time.time()
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        dur = time.time() - start
        prec = precision_score(yte, y_pred, average="weighted", zero_division=0)
        nmi = normalized_mutual_info_score(yte, y_pred)
        ari = adjusted_rand_score(yte, y_pred)
        return prec, nmi, ari, dur

    print("\n=== Holdout Test (80/20 split, random_state=42) ===")
    results = {}
    for name, model in models_ordered.items():
        prec, nmi, ari, dur = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        results[name] = (prec, nmi, ari, dur)
        print(f"{name:<20} prec={prec:.4f} nmi={nmi:.4f} ari={ari:.4f} time={dur:.1f}s")

    print("\n=== 5-Fold Stratified Cross-Validation ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = {
        "XGBoost": Pipeline([("scaler", StandardScaler()), ("clf", make_xgb(200))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", random_state=42))]),
        "Random Forest": Pipeline(
            [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2))]
        ),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42, n_jobs=1, solver="lbfgs"))]
        ),
    }

    scoring = ["precision_weighted", "f1_weighted", "accuracy"]
    detailed_results = []
    X = omics_data_combined
    y = labels

    for name, pipeline in pipelines.items():
        print(f"Running 5-fold CV for {name}...")
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=1)
        row = {
            "Model": name,
            "Precision_Mean": scores["test_precision_weighted"].mean(),
            "Precision_Std": scores["test_precision_weighted"].std(),
            "F1_Mean": scores["test_f1_weighted"].mean(),
            "F1_Std": scores["test_f1_weighted"].std(),
            "Accuracy_Mean": scores["test_accuracy"].mean(),
            "Accuracy_Std": scores["test_accuracy"].std(),
        }
        detailed_results.append(row)
        print(
            f"{name:<20} prec={row['Precision_Mean']:.4f} f1={row['F1_Mean']:.4f} "
            f"acc={row['Accuracy_Mean']:.4f} std={row['Precision_Std']:.4f}"
        )

    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv("5fold_cv_results.csv", index=False)
    print("\nSaved 5fold_cv_results.csv")

    print("\n=== Confusion Matrices (cross_val_predict) ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    for idx, (name, pipeline) in enumerate(pipelines.items()):
        y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)
        cm = confusion_matrix(y, y_pred_cv)
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=axes[idx])
        axes[idx].set_title(f"Confusion Matrix - {name}")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        report = classification_report(y, y_pred_cv, output_dict=True)
        wa = report["weighted avg"]
        print(
            f"{name:<20} weighted precision={wa['precision']:.4f} recall={wa['recall']:.4f} f1={wa['f1-score']:.4f}"
        )
    plt.tight_layout()
    plt.savefig("confusion_matrices_5fold.png", dpi=300, bbox_inches="tight")
    print("Saved confusion_matrices_5fold.png")

    print("\n=== Consistency Analysis ===")
    for row in detailed_results:
        consistency = 1 - (row["Precision_Std"] / row["Precision_Mean"])
        cv_percent = row["Precision_Std"] / row["Precision_Mean"] * 100
        print(f"{row['Model']:<20} consistency={consistency:.4f} (precision CV {cv_percent:.2f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()