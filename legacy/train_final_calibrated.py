import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

CSV = "data/processed/all_match_features.csv"
OUT_JOBLIB = "models/xgb_60s_final_calibrated.joblib"
OUT_META = "models/xgb_60s_final_calibrated_metadata.txt"

print("Loading data...")
df = pd.read_csv(CSV)
pipe = joblib.load("models/xgb_60s_calibrated.joblib")
feature_cols = pipe["feature_cols"]

X = df[feature_cols].astype(float)
y = df["win"].astype(int)

print(f"Rows: {len(df)} | Features: {feature_cols}")

base = XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)

print("Fitting calibrated classifier on all data (this may take a minute)...")
cal = CalibratedClassifierCV(base, cv=5, method="sigmoid")
cal.fit(X, y)

print("Computing in-sample probabilities...")
probs_in = cal.predict_proba(X)[:, 1]

print("Computing out-of-fold (OOF) calibrated probabilities for reference (may take longer)...")
oof_probs = cross_val_predict(CalibratedClassifierCV(base, cv=5, method="sigmoid"),
                              X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
print("In-sample Accuracy:", accuracy_score(y, (probs_in >= 0.5).astype(int)))
print("In-sample AUC:    ", roc_auc_score(y, probs_in))
print("In-sample Brier:  ", brier_score_loss(y, probs_in))
print("OOF Accuracy:     ", accuracy_score(y, (oof_probs >= 0.5).astype(int)))
print("OOF AUC:          ", roc_auc_score(y, oof_probs))
print("OOF Brier:        ", brier_score_loss(y, oof_probs))

# Save pipeline
joblib.dump({"checkpoint_seconds": 60, "feature_cols": feature_cols, "pipeline": cal}, OUT_JOBLIB)
with open(OUT_META, "w", encoding="utf-8") as fh:
    fh.write(f"checkpoint_seconds=60\n")
    fh.write("feature_cols=" + ",".join(feature_cols) + "\n")
    fh.write(f"in_sample_auc={roc_auc_score(y, probs_in):.6f}\n")
    fh.write(f"oof_auc={roc_auc_score(y, oof_probs):.6f}\n")

print("Saved final calibrated pipeline to:", OUT_JOBLIB)
print("Saved metadata to:", OUT_META)