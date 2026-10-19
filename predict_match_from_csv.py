import sys, joblib, pandas as pd

MODEL_PATH = "models/xgb_60s_final_calibrated.joblib"
CSV = "data/processed/all_match_features.csv"

pipe = joblib.load(MODEL_PATH)
feature_cols = pipe["feature_cols"]
cal = pipe["pipeline"]

df = pd.read_csv(CSV)

if len(sys.argv) > 1:
    match = sys.argv[1]
    row = df[df["match_file"] == match]
    if row.empty:
        print("Match not found in CSV:", match)
        sys.exit(1)
    X = row[feature_cols].astype(float)
    prob = cal.predict_proba(X)[:, 1][0]
    print(f"{match}  P(win | 60s) = {prob:.4f}")
else:
    # write predictions for all matches
    X = df[feature_cols].astype(float)
    probs = cal.predict_proba(X)[:, 1]
    out = df[["match_file"]].copy()
    out["p_win_60s"] = probs
    out.to_csv("data/processed/all_match_predictions_60s.csv", index=False)
    print("Wrote predictions for", len(out), "matches to data/processed/all_match_predictions_60s.csv")