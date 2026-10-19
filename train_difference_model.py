import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

df = pd.read_csv("sc2_paired_training.csv")

feature_names = [
    c.replace("player1_", "")
    for c in df.columns
    if c.startswith("player1_") and c != "player1_win"
    and c.replace("player1_", "player2_") in df.columns
]

# Each feature describes Player 1's advantage over Player 2.
X = pd.DataFrame({
    name: df["player1_" + name] - df["player2_" + name]
    for name in feature_names
})
y = df["player1_win"].astype(int)

# Split before augmentation to avoid test-set leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Mirror TRAINING matches only: reverse the advantage and winner.
X_mirror = -X_train
y_mirror = 1 - y_train

X_aug = pd.concat([X_train, X_mirror], ignore_index=True)
y_aug = pd.concat([y_train, y_mirror], ignore_index=True)

models = {
    "Difference Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),
    "Difference Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
}

print("Original training matches:", len(X_train))
print("Augmented training rows:", len(X_aug))
print("Untouched test matches:", len(X_test))
print("Difference features:", len(feature_names))

results = []

for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_aug, y_aug)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    balanced = balanced_accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced accuracy": balanced,
        "ROC AUC": auc
    })

    print("Accuracy:", round(accuracy, 4))
    print("Balanced accuracy:", round(balanced, 4))
    print("ROC AUC:", round(auc, 4))

print("\n=== DIFFERENCE MODEL COMPARISON ===")
print(pd.DataFrame(results).to_string(index=False))

print("\nExperiment complete. No existing model files were changed.")
