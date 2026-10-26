import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier

df = pd.read_csv("sc2_paired_training.csv")

X = df.drop(columns=["replay_index", "player1_win"])
y = df["player1_win"].astype(int)

constant_cols = X.columns[X.nunique(dropna=False) <= 1]
X = X.drop(columns=constant_cols)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    )
}

print("Rows:", len(df))
print("Features:", X.shape[1])
print("Validation: 5-fold stratified cross-validation")

for name, model in models.items():
    print(f"\nEvaluating {name}...", flush=True)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc"
        },
        n_jobs=1
    )

    for metric in ["accuracy", "balanced_accuracy", "roc_auc"]:
        values = scores["test_" + metric]
        print(
            f"{metric}: {values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )
        print("  Fold scores:", np.round(values, 4))

print("\nCross-validation complete.")
