import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
import joblib

training = pd.read_csv("sc2_paired_training.csv")
metadata = pd.read_csv("replay_race_map_metadata.csv")

df = training.merge(
    metadata[["replay_index", "player1_race", "player2_race"]],
    on="replay_index",
    how="inner",
    validate="one_to_one",
)

numeric = [
    c for c in training.columns
    if c.startswith("player1_scoreValue")
    or c.startswith("player2_scoreValue")
]
features = numeric + ["player1_race", "player2_race"]

X = df[features]
y = df["player1_win"].astype(int)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X, y)
baseline_predictions = baseline.predict(X_test)
baseline_probabilities = baseline.predict_proba(X_test)[:, 1]

model = joblib.load("final_sc2_winner_model.joblib")
model_predictions = model.predict(X_test)
model_probabilities = model.predict_proba(X_test)[:, 1]

print("=== SAME TEST SET COMPARISON ===")
for name, pred, prob in [
    ("Majority baseline", baseline_predictions, baseline_probabilities),
    ("Logistic regression", model_predictions, model_probabilities),
]:
    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(y_test, pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.4f}")
    print(f"ROC AUC:           {roc_auc_score(y_test, prob):.4f}")
