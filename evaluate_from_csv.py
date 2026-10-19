import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

CSV_PATH = "data/processed/all_match_features.csv"
OUT_CSV = "data/processed/time_horizon_metrics.csv"
CHECKPOINTS = [60, 120, 180, 240, 300, 360]

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(CSV_PATH)

df = pd.read_csv(CSV_PATH)
print("Loaded", len(df), "matches from", CSV_PATH)
print("Win distribution:\n", df["win"].value_counts())

metrics = []
for sec in CHECKPOINTS:
    feature_cols = [
        f"p1_apm@{sec}",
        f"p1_workers@{sec}",
        f"p1_buildings@{sec}",
        f"delta_apm@{sec}",
        f"delta_workers@{sec}",
        f"delta_buildings@{sec}",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Skipping {sec}s — missing columns: {missing}")
        metrics.append({
            "checkpoint_seconds": sec,
            "acc_mean": np.nan, "acc_std": np.nan,
            "auc_mean": np.nan, "auc_std": np.nan,
            "brier_mean": np.nan, "brier_std": np.nan,
        })
        continue

    X = df[feature_cols].fillna(0).astype(float)
    y = df["win"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, briers = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            gamma=0,
            min_child_weight=1,
            random_state=42,
            use_label_encoder=False,
            verbosity=0
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]

        accs.append(accuracy_score(y.iloc[test_idx], preds))
        aucs.append(roc_auc_score(y.iloc[test_idx], probs))
        briers.append(brier_score_loss(y.iloc[test_idx], probs))

    metrics.append({
        "checkpoint_seconds": sec,
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
    })
    print(f"Done {sec}s -> acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f} brier={np.mean(briers):.4f}")

out = pd.DataFrame(metrics)
out.to_csv(OUT_CSV, index=False)
print("\n=== Time-Horizon CV Summary ===")
print(out.to_string(index=False))
print(f"\nSaved metrics to {OUT_CSV}")