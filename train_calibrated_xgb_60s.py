import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

CSV_PATH = "data/processed/all_match_features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Use strongest checkpoint according to the evaluation
CHECKPOINT = 60
TARGET = "win"

feature_cols = [
    f"p1_apm@{CHECKPOINT}",
    f"p1_workers@{CHECKPOINT}",
    f"p1_buildings@{CHECKPOINT}",
    f"delta_apm@{CHECKPOINT}",
    f"delta_workers@{CHECKPOINT}",
    f"delta_buildings@{CHECKPOINT}",
]

df = pd.read_csv(CSV_PATH)
print("Rows:", len(df))
print("Win distribution:\n", df[TARGET].value_counts())

# Keep only valid columns
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns for checkpoint {CHECKPOINT}: {missing}")

X = df[feature_cols].fillna(0).astype(float)
y = df[TARGET].astype(int)

# Train / validation / test split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Tuned model from the earlier search
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

print("\nTraining base XGBoost...")
base_model.fit(X_train, y_train)

# Get raw probabilities on validation
val_probs = base_model.predict_proba(X_val)[:, 1]
test_probs = base_model.predict_proba(X_test)[:, 1]

# Fit sigmoid calibrator on validation set
calibrator = LogisticRegression(
    random_state=42,
    max_iter=1000
)
calibrator.fit(val_probs.reshape(-1, 1), y_val)

# Use calibration on test
calibrated_test_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]

# Final metrics
acc = accuracy_score(y_test, (calibrated_test_probs >= 0.5).astype(int))
auc = roc_auc_score(y_test, calibrated_test_probs)
brier = brier_score_loss(y_test, calibrated_test_probs)

print("\n=== Final test metrics (60s, calibrated) ===")
print(f"Accuracy: {acc:.4f}")
print(f"AUC:     {auc:.4f}")
print(f"Brier:   {brier:.4f}")

# Save final pipeline artifacts
save_path = os.path.join(MODEL_DIR, "xgb_60s_calibrated.joblib")
joblib.dump({
    "checkpoint_seconds": CHECKPOINT,
    "feature_cols": feature_cols,
    "base_model": base_model,
    "calibrator": calibrator
}, save_path)

print(f"\nSaved final model pipeline to: {save_path}")

# Save a simple metadata file
meta_path = os.path.join(MODEL_DIR, "xgb_60s_calibrated_metadata.txt")
with open(meta_path, "w", encoding="utf-8") as f:
    f.write(f"checkpoint_seconds={CHECKPOINT}\n")
    f.write("feature_cols=" + ",".join(feature_cols) + "\n")
    f.write(f"accuracy={acc:.6f}\n")
    f.write(f"auc={auc:.6f}\n")
    f.write(f"brier={brier:.6f}\n")

print(f"Saved metadata to: {meta_path}")