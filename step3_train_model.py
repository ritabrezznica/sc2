import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib

DATA_PATH = "./data/processed/real_sc2egset_10k_matches.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Real parsed dataset not found at {DATA_PATH}. Run extract_real_dataset.py first!")

print(f"--> Loading Real Empirical SC2EGSet Dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Output true dataset properties
total_matches = len(df)
p1_wins = df['win'].sum()
p1_win_rate = (p1_wins / total_matches) * 100

print(f"\n================ EMPIRICAL DATASET METADATA ================")
print(f"Total Valid Matches (N) : {total_matches}")
print(f"Player 1 Win Distribution: {p1_win_rate:.2f}% Win ({p1_wins}) | {100 - p1_win_rate:.2f}% Loss ({total_matches - p1_wins})")
print("============================================================\n")

# Isolate Real Feature Vectors
X_frame_a = df[['p1_apm', 'p1_workers', 'p1_buildings', 'p1_army', 'p1_unspent']]
X_frame_b = df[['delta_apm', 'delta_workers', 'delta_buildings', 'delta_army', 'delta_unspent']]
y = df['win']

# 10-Fold Stratified Cross-Validation (Match-Level Partitioning)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'Majority Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
rf_fold_acc_a, rf_fold_acc_b = [], []

print("--> Executing Multi-Model 10-Fold Stratified Evaluation on Real Telemetry...")

for name, model in models.items():
    if name == 'Majority Baseline':
        accs = []
        for _, test_idx in skf.split(X_frame_a, y):
            model.fit(X_frame_a.iloc[:10], y.iloc[:10]) # Dummy fit
            preds = model.predict(X_frame_a.iloc[test_idx])
            accs.append(accuracy_score(y.iloc[test_idx], preds))
            
        results.append({
            'Model Architecture': name,
            'Frame A Acc (%)': f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
            'Frame B Acc (%)': f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
            'Frame A ROC-AUC': "0.5000 ± 0.0000",
            'Frame B ROC-AUC': "0.5000 ± 0.0000",
            'Acc Diff (pp)': "0.00 pp"
        })
        continue

    # Evaluate Frame A (Absolute Real Player State)
    accs_a, aucs_a = [], []
    for train_idx, test_idx in skf.split(X_frame_a, y):
        model.fit(X_frame_a.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_frame_a.iloc[test_idx])
        probas = model.predict_proba(X_frame_a.iloc[test_idx])[:, 1]
        accs_a.append(accuracy_score(y.iloc[test_idx], preds))
        aucs_a.append(roc_auc_score(y.iloc[test_idx], probas))
        if name == 'Random Forest': rf_fold_acc_a.append(accuracy_score(y.iloc[test_idx], preds))

    # Evaluate Frame B (Opponent-Relative Differential)
    accs_b, aucs_b = [], []
    for train_idx, test_idx in skf.split(X_frame_b, y):
        model.fit(X_frame_b.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_frame_b.iloc[test_idx])
        probas = model.predict_proba(X_frame_b.iloc[test_idx])[:, 1]
        accs_b.append(accuracy_score(y.iloc[test_idx], preds))
        aucs_b.append(roc_auc_score(y.iloc[test_idx], probas))
        if name == 'Random Forest': rf_fold_acc_b.append(accuracy_score(y.iloc[test_idx], preds))

    results.append({
        'Model Architecture': name,
        'Frame A Acc (%)': f"{np.mean(accs_a)*100:.2f} ± {np.std(accs_a)*100:.2f}",
        'Frame B Acc (%)': f"{np.mean(accs_b)*100:.2f} ± {np.std(accs_b)*100:.2f}",
        'Frame A ROC-AUC': f"{np.mean(aucs_a):.4f} ± {np.std(aucs_a):.4f}",
        'Frame B ROC-AUC': f"{np.mean(aucs_b):.4f} ± {np.std(aucs_b):.4f}",
        'Acc Diff (pp)': f"{(np.mean(accs_b) - np.mean(accs_a))*100:+.2f} pp"
    })

df_res = pd.DataFrame(results)
print("\n================ EMPIRICAL MULTI-MODEL EXPERIMENTAL MATRIX ================")
print(df_res.to_string(index=False))
print("=========================================================================\n")

# Exploratory Fold-Level Paired Analysis
t_stat, p_val = stats.ttest_rel(rf_fold_acc_b, rf_fold_acc_a)
print("================ EXPLORATORY STATISTICAL FOLD ANALYSIS ================")
print(f"Paired Two-Tailed t-test across 10 CV folds (Random Forest):")
print(f"  t-statistic = {t_stat:.4f} | p-value = {p_val:.6f}")
print("  Note: CV folds share training data; interpret p-values as an exploratory indicator of consistent fold-level performance gains.")
print("=======================================================================\n")

# Retrain Final Deployment Models on Complete Dataset for Streamlit Prototype
print("--> Retraining final deployment models on complete dataset for Streamlit prototype...")
rf_a = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_frame_a, y)
rf_b = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_frame_b, y)

os.makedirs("./models", exist_ok=True)
joblib.dump(rf_a, "./models/sc2_rf_frame_a.pkl")
joblib.dump(rf_b, "./models/sc2_rf_frame_b.pkl")
print("[SUCCESS] Trained models saved to ./models/")