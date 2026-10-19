import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load the paired dataset
df = pd.read_csv("sc2_paired_training.csv")

target = "player1_win"
X = df.drop(columns=["replay_index", target])
y = df[target].astype(int)

# 2. Basic validation
assert len(df) == 23423, f"Unexpected row count: {len(df)}"
assert set(y.unique()) == {0, 1}, "Target must contain 0 and 1"
assert not X.isin([np.inf, -np.inf]).any().any(), "Infinite feature values found"

print("Dataset shape:", df.shape)
print("Feature count:", X.shape[1])
print("Target distribution:")
print(y.value_counts().sort_index().rename(index={0: "Player 1 loses", 1: "Player 1 wins"}))

# 3. Split by replay into training and test sets
# Each row represents one complete replay matchup.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining matches:", len(X_train))
print("Test matches:", len(X_test))

# 4. Define models
models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}

# 5. Train and evaluate
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = float("nan")

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced accuracy": balanced,
        "ROC AUC": auc
    })
    trained_models[name] = model

    print("Accuracy:", round(accuracy, 4))
    print("Balanced accuracy:", round(balanced, 4))
    print("ROC AUC:", round(auc, 4) if not np.isnan(auc) else "N/A")

# 6. Compare models
results_df = pd.DataFrame(results).sort_values(
    "ROC AUC", ascending=False, na_position="last"
)

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

# 7. Save the best trained model with probabilities
eligible = results_df[results_df["ROC AUC"].notna()]
best_name = eligible.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(
    {
        "model": best_model,
        "feature_columns": list(X.columns),
        "target": target,
        "model_name": best_name
    },
    "sc2_winner_model.joblib"
)

print("\n=== BEST MODEL TEST REPORT ===")
print("Selected by test ROC AUC:", best_name)
best_predictions = best_model.predict(X_test)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, best_predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    target_names=["Player 1 loses", "Player 1 wins"]
))

results_df.to_csv("model_comparison.csv", index=False)

print("\nSaved model: sc2_winner_model.joblib")
print("Saved comparison: model_comparison.csv")
print("\nPhase 4 initial training complete.")
