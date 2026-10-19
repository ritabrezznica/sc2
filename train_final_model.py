import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

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
categorical = ["player1_race", "player2_race"]
features = numeric + categorical

X = df[features]
y = df["player1_win"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

preprocessing = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical,
    ),
])

model = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", LogisticRegression(max_iter=3000)),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("=== FINAL MODEL TEST RESULTS ===")
print("Total matches:", len(df))
print("Training matches:", len(X_train))
print("Test matches:", len(X_test))
print("Numeric features:", len(numeric))
print("Categorical features:", categorical)

print(f"\nAccuracy:          {accuracy_score(y_test, predictions):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
print(f"ROC AUC:           {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Player 1 loses", "Player 1 wins"],
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "final_sc2_winner_model.joblib")
print("\nSaved model: final_sc2_winner_model.joblib")
