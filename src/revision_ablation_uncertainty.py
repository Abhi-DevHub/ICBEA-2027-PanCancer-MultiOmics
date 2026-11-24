# -*- coding: utf-8 -*-
"""AUDIT 5+6 — uncertainty for leave-one-modality-out ablation differences.

Runs 5 repeats x 5 folds (25 paired folds) within the 80% TRAINING partition for
each modality configuration, then computes paired differences (full - config)
with Nadeau-Bengio corrected CI and p-values, so we can say whether removing a
modality genuinely changes performance.

Output: revision_ablation_uncertainty.csv, revision_ablation_perfold.csv
"""
import sys, time, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from common import load_data, make_xgb

R = r"D:\Test_XGboost\XGBoost_Run" + "\\"
REPEATS, FOLDS = 5, 5
J = REPEATS * FOLDS
DF = J - 1
NTT = 1.0 / (FOLDS - 1)     # test/train size ratio for 5-fold = 1/4

print("Loading data...")
X, y = load_data(None)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

suffixes = {"mRNA": "_mRNA", "miRNA": "_miRNA", "CNV": "_CNV", "Methy": "_Methy"}
cols = {k: [c for c in X.columns if c.endswith(s)] for k, s in suffixes.items()}
setups = [("full", list(X.columns))]
for drop, c in cols.items():
    setups.append((f"without_{drop}", [x for x in X.columns if x not in set(c)]))
setups.append(("mRNA_only", cols["mRNA"]))

per_fold = []
t0 = time.time()
for name, feat in setups:
    Xs = X_train[feat]
    print(f"\n=== {name} ({len(feat)} features) ===")
    for rep in range(REPEATS):
        seed = 1000 * (rep + 1)
        kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(Xs, y_train)):
            sc = StandardScaler().fit(Xs.iloc[tr])
            m = make_xgb(200)
            m.fit(sc.transform(Xs.iloc[tr]), y_train.iloc[tr])
            pred = m.predict(sc.transform(Xs.iloc[te]))
            per_fold.append({
                "config": name, "repeat": rep, "fold": fold,
                "accuracy": accuracy_score(y_train.iloc[te], pred),
                "macro_f1": f1_score(y_train.iloc[te], pred, average="macro", zero_division=0),
            })
        print(f"  rep={rep} done [{time.time()-t0:.0f}s]")

PF = pd.DataFrame(per_fold)
PF.to_csv(R + "revision_ablation_perfold.csv", index=False)
print(f"\nsaved revision_ablation_perfold.csv ({len(PF)} rows)")

# ---- paired deltas vs full ----
rows = []
for metric in ["accuracy", "macro_f1"]:
    base = PF[PF.config == "full"].set_index(["repeat", "fold"])[metric]
    for name, _ in setups[1:]:
        cur = PF[PF.config == name].set_index(["repeat", "fold"])[metric]
        d = (base - cur).reindex(base.index).values   # full - without_X
        n = len(d); mean = d.mean(); sd = d.std(ddof=1)
        se_nb = math.sqrt((1.0 / J + NTT) * sd ** 2)
        t_nb = mean / se_nb if se_nb > 0 else np.nan
        ci = (mean - stats.t.ppf(0.975, DF) * se_nb, mean + stats.t.ppf(0.975, DF) * se_nb)
        p2 = 2 * stats.t.sf(abs(t_nb), DF) if np.isfinite(t_nb) else np.nan
        wr = stats.wilcoxon(base.values, cur.reindex(base.index).values)
        rows.append({
            "comparison": f"full - {name}",
            "metric": metric, "n_paired": n,
            "mean_delta": round(float(mean), 5),
            "sd_delta": round(float(sd), 5),
            "ci95_NB": f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
            "cohens_d": round(float(mean / sd), 3) if sd > 0 else np.nan,
            "p_NB_2s": round(float(p2), 5),
            "wilcoxon_p": round(float(wr.pvalue), 5),
            "significant_5pct": "YES" if (np.isfinite(p2) and p2 < 0.05) else "no",
        })

U = pd.DataFrame(rows)
U.to_csv(R + "revision_ablation_uncertainty.csv", index=False)
print("\n" + "=" * 100)
print(U.to_string(index=False))
print("=" * 100)
print(f"\nDONE in {time.time()-t0:.0f}s")
