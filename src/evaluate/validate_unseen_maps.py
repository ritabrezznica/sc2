import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate

training = pd.read_csv("sc2_paired_training.csv")
metadata = pd.read_csv("replay_race_map_metadata.csv")

df = training.merge(
    metadata[[
        "replay_index", "player1_race", "player2_race", "map_name"
    ]],
    on="replay_index",
    how="inner",
    validate="one_to_one",
)

df["race_matchup"] = (
    df["player1_race"] + "_vs_" + df["player2_race"]
)

numeric = [
    c for c in training.columns
    if c.startswith("player1_scoreValue")
    or c.startswith("player2_scoreValue")
]

target = df["player1_win"].astype(int)
groups = df["map_name"]

experiments = {
    "Statistics only": numeric,
    "Statistics + race": numeric + [
        "player1_race", "player2_race"
    ],
    "Statistics + race + matchup": numeric + [
        "player1_race", "player2_race", "race_matchup"
    ],
}

cv = GroupKFold(n_splits=5)

print("Matches:", len(df))
print("Unique maps:", groups.nunique())
print("Validation: 5-fold GroupKFold grouped by map")
print("Each validation fold contains maps excluded from its training fold.")

results = []

for name, features in experiments.items():
    categorical = [c for c in features if c not in numeric]

    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=3000)),
    ])

    scores = cross_validate(
        model,
        df[features],
        target,
        groups=groups,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "roc_auc"],
        n_jobs=-1,
        error_score="raise",
    )

    row = {"experiment": name}

    for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
        vals = scores[f"test_{metric}"]
        row[metric] = vals.mean()
        row[metric + "_std"] = vals.std()

    results.append(row)

    print(f"\n{name}")
    print(f"  Accuracy:          {row['accuracy']:.4f} +/- {row['accuracy_std']:.4f}")
    print(f"  Balanced accuracy: {row['balanced_accuracy']:.4f} +/- {row['balanced_accuracy_std']:.4f}")
    print(f"  ROC AUC:           {row['roc_auc']:.4f} +/- {row['roc_auc_std']:.4f}")

out = pd.DataFrame(results)
out.to_csv("unseen_map_validation.csv", index=False)

print("\nSaved: unseen_map_validation.csv")
