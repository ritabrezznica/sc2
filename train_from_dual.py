"""
Reproducible training script for dual-frame dataset.
Saves: sc2_rf_frame_a_retrain.pkl, sc2_rf_frame_b_retrain.pkl
Also saves evaluation CSV: reports/dual_frame_cv_results.csv and reports/predictions.csv
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss, precision_recall_fscore_support

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "data/processed/dual_frame_match_features.csv"
OUT_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load
df = pd.read_csv(DATA_PATH)
print("Loaded", len(df), "rows from", DATA_PATH)

# Features and labels
X_a = df[['p1_apm','p1_workers','p1_buildings','p1_army','p1_unspent']].copy()
X_b = df[['delta_apm','delta_workers','delta_buildings','delta_army','delta_unspent']].copy()
y = df['win'].astype(int).copy()

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def evaluate_and_save(X, y, name):
    print(f"Evaluating: {name}")
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    # cross_val_predict with predict_proba to get out-of-fold probabilities
    probas = cross_val_predict(clf, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
    preds = (probas >= 0.5).astype(int)
    # metrics
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probas)
    f1 = f1_score(y, preds)
    brier = brier_score_loss(y, probas)
    prec, rec, f1s, _ = precision_recall_fscore_support(y, preds, average='binary')
    print(f"{name} — acc={acc:.4f} auc={auc:.4f} f1={f1:.4f} brier={brier:.4f}")
    # save per-sample predictions for calibration analysis
    results_df = X.copy()
    results_df['y_true'] = y
    results_df['y_pred_proba'] = probas
    results_df['y_pred'] = preds
    results_df.to_csv(os.path.join(REPORT_DIR, f"predictions_{name}.csv"), index=False)
    # retrain on full data and save final model
    clf.fit(X, y)
    model_path = os.path.join(OUT_DIR, f"sc2_rf_{name}_retrain.pkl")
    joblib.dump(clf, model_path)
    meta = {
        "model": "RandomForestClassifier",
        "n_estimators": clf.n_estimators,
        "random_state": RANDOM_SEED,
        "features": list(X.columns),
        "rows": len(X)
    }
    with open(os.path.join(OUT_DIR, f"sc2_rf_{name}_retrain_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model:", model_path)
    return {"name": name, "acc": acc, "auc": auc, "f1": f1, "brier": brier, "rows": len(X)}

results = []
results.append(evaluate_and_save(X_a, y, "frame_a"))
results.append(evaluate_and_save(X_b, y, "frame_b"))

pd.DataFrame(results).to_csv(os.path.join(REPORT_DIR, "dual_frame_cv_results.csv"), index=False)
print("Saved CV summary to reports/dual_frame_cv_results.csv")