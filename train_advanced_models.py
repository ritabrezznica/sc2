import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load verified paired dataset
df = pd.read_csv("sc2_paired_training.csv")

target = "player1_win"
X = df.drop(columns=["replay_index", target])
y = df[target].astype(int)

# Remove constant features only
constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
X = X.drop(columns=constant_cols)

print("Dataset rows:", len(df))
print("Original features:", 78)
print("Constant features removed:", len(constant_cols))
print("Remaining features:", X.shape[1])
print("Removed feature examples:", constant_cols[:10])

# Same split as previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=15,
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.03,
        verbose=False,
        random_seed=42,
        thread_count=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...", flush=True)
    model.fit(X_train, y_train)

    pred = model.predict(X_test).ravel().astype(int)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
    else:
        auc = float("nan")

    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced accuracy": bal,
        "ROC AUC": auc
    })

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal:.4f}")
    print(f"ROC AUC: {auc:.4f}" if not np.isnan(auc) else "ROC AUC: N/A")

results_df = pd.DataFrame(results).sort_values(
    "ROC AUC", ascending=False, na_position="last"
)

print("\n=== FINAL COMPARISON ===")
print(results_df.to_string(index=False))

results_df.to_csv("advanced_model_comparison.csv", index=False)
print("\nSaved: advanced_model_comparison.csv")
print("Existing saved model was not changed.")
