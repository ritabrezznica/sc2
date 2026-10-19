import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

CSV_PATH = "data/processed/all_match_features.csv"
OUT_CSV = "data/processed/all_match_features_rich.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT = 60
TARGET = "win"

def add_rich_features(df, sec):
    base = df.copy()

    # Opponent-side reconstruction from existing deltas
    # Assumes delta_x = p1_x - p2_x
    for side in ["apm", "workers", "buildings"]:
        p1_col = f"p1_{side}@{sec}"
        delta_col = f"delta_{side}@{sec}"
        if p1_col in base.columns and delta_col in base.columns:
            base[f"p2_{side}@{sec}"] = base[p1_col] - base[delta_col]

    # Resource proxies: use worker/building/apm as a macro signal
    # These are not raw spectator values, but they are very useful derived features.
    if f"p1_workers@{sec}" in base.columns:
        base[f"p1_minerals@{sec}"] = (
            10.0 * base[f"p1_workers@{sec}"] +
            22.0 * base[f"p1_buildings@{sec}"] +
            0.2 * base[f"p1_apm@{sec}"]
        )
        base[f"p2_minerals@{sec}"] = (
            10.0 * base[f"p2_workers@{sec}"] +
            22.0 * base[f"p2_buildings@{sec}"] +
            0.2 * base[f"p2_apm@{sec}"]
        )
        base[f"p1_vespene@{sec}"] = (
            6.0 * base[f"p1_workers@{sec}"] +
            12.0 * base[f"p1_buildings@{sec}"] +
            0.15 * base[f"p1_apm@{sec}"]
        )
        base[f"p2_vespene@{sec}"] = (
            6.0 * base[f"p2_workers@{sec}"] +
            12.0 * base[f"p2_buildings@{sec}"] +
            0.15 * base[f"p2_apm@{sec}"]
        )

    # Supply / army proxies
    if f"p1_workers@{sec}" in base.columns and f"p1_buildings@{sec}" in base.columns:
        base[f"p1_supply@{sec}"] = base[f"p1_workers@{sec}"] + 3.0 * base[f"p1_buildings@{sec}"]
        base[f"p2_supply@{sec}"] = base[f"p2_workers@{sec}"] + 3.0 * base[f"p2_buildings@{sec}"]

        base[f"p1_army_value@{sec}"] = (
            1.5 * base[f"p1_workers@{sec}"] +
            4.0 * base[f"p1_buildings@{sec}"] +
            0.08 * base[f"p1_apm@{sec}"]
        )
        base[f"p2_army_value@{sec}"] = (
            1.5 * base[f"p2_workers@{sec}"] +
            4.0 * base[f"p2_buildings@{sec}"] +
            0.08 * base[f"p2_apm@{sec}"]
        )

    # Macro balance signals
    if f"p1_apm@{sec}" in base.columns and f"p2_apm@{sec}" in base.columns:
        base[f"delta_minerals@{sec}"] = base[f"p1_minerals@{sec}"] - base[f"p2_minerals@{sec}"]
        base[f"delta_vespene@{sec}"] = base[f"p1_vespene@{sec}"] - base[f"p2_vespene@{sec}"]
        base[f"delta_supply@{sec}"] = base[f"p1_supply@{sec}"] - base[f"p2_supply@{sec}"]
        base[f"delta_army_value@{sec}"] = base[f"p1_army_value@{sec}"] - base[f"p2_army_value@{sec}"]

    return base

# Load base CSV
df = pd.read_csv(CSV_PATH)
df = add_rich_features(df, CHECKPOINT)

# Save richer features
df.to_csv(OUT_CSV, index=False)
print(f"Saved rich feature CSV to: {OUT_CSV}")
print(df.head().to_string(index=False))

# Feature selection: start from original features + richer additions
base_features = [
    "p1_apm@60",
    "p1_workers@60",
    "p1_buildings@60",
    "delta_apm@60",
    "delta_workers@60",
    "delta_buildings@60",
]

rich_features = [
    "p2_apm@60",
    "p2_workers@60",
    "p2_buildings@60",
    "p1_minerals@60",
    "p2_minerals@60",
    "p1_vespene@60",
    "p2_vespene@60",
    "p1_supply@60",
    "p2_supply@60",
    "p1_army_value@60",
    "p2_army_value@60",
    "delta_minerals@60",
    "delta_vespene@60",
    "delta_supply@60",
    "delta_army_value@60",
]

feature_cols = [c for c in base_features + rich_features if c in df.columns]
X = df[feature_cols].fillna(0).astype(float)
y = df[TARGET].astype(int)

# Train/Val/Test split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

base_model = XGBClassifier(
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

base_model.fit(X_train, y_train)

val_probs = base_model.predict_proba(X_val)[:, 1]
calibrator = LogisticRegression(random_state=42, max_iter=1000)
calibrator.fit(val_probs.reshape(-1, 1), y_val)

test_probs = base_model.predict_proba(X_test)[:, 1]
cal_test_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]

acc = accuracy_score(y_test, (cal_test_probs >= 0.5).astype(int))
auc = roc_auc_score(y_test, cal_test_probs)
brier = brier_score_loss(y_test, cal_test_probs)

print("\n=== Rich-features 60s test metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Brier: {brier:.4f}")
print(f"Feature count: {len(feature_cols)}")

joblib.dump({
    "checkpoint_seconds": CHECKPOINT,
    "feature_cols": feature_cols,
    "base_model": base_model,
    "calibrator": calibrator
}, os.path.join(MODEL_DIR, "xgb_60s_rich_calibrated.joblib"))

with open(os.path.join(MODEL_DIR, "xgb_60s_rich_calibrated_metadata.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"checkpoint_seconds={CHECKPOINT}\n")
    fh.write("feature_cols=" + ",".join(feature_cols) + "\n")
    fh.write(f"accuracy={acc:.6f}\n")
    fh.write(f"auc={auc:.6f}\n")
    fh.write(f"brier={brier:.6f}\n")

print(f"\nSaved rich model to: {MODEL_DIR}/xgb_60s_rich_calibrated.joblib")
print(f"Saved rich metadata to: {MODEL_DIR}/xgb_60s_rich_calibrated_metadata.txt")