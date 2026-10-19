
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

TRAINING_PATH = "sc2_paired_training.csv"
IDENTITY_PATH = "replay_player_identity_metadata.csv"
RESULTS_PATH = "unseen_player_stability_results.csv"

# Ten different player-identity splits.
RANDOM_SEEDS = [11, 21, 31, 41, 42, 51, 61, 71, 81, 91]

# --------------------------------------------------
# 1. Load and merge validated data
# --------------------------------------------------

training = pd.read_csv(TRAINING_PATH, encoding="utf-8-sig")
identity = pd.read_csv(IDENTITY_PATH, encoding="utf-8-sig")

data = training.merge(
    identity[
        [
            "replay_index",
            "player1_key",
            "player2_key",
            "player1_race",
            "player2_race",
        ]
    ],
    on="replay_index",
    how="inner",
    validate="one_to_one",
)

# Require both player identities and both race values.
data = data.dropna(
    subset=[
        "player1_key",
        "player2_key",
        "player1_race",
        "player2_race",
    ]
).copy()

for column in ["player1_key", "player2_key"]:
    data[column] = data[column].astype(str).str.strip()

data = data[
    data["player1_key"].ne("")
    & data["player2_key"].ne("")
    & data["player1_key"].str.lower().ne("nan")
    & data["player2_key"].str.lower().ne("nan")
].copy()

data["player1_win"] = data["player1_win"].astype(int)

player_keys = pd.unique(
    pd.concat(
        [data["player1_key"], data["player2_key"]],
        ignore_index=True,
    )
)

excluded_columns = {
    "replay_index",
    "player1_win",
    "player1_key",
    "player2_key",
}

numeric_features = [
    column
    for column in training.columns
    if column not in excluded_columns
]

categorical_features = ["player1_race", "player2_race"]

print("=== UNSEEN-PLAYER STABILITY TEST ===")
print("Valid matches:", len(data))
print("Unique player identities:", len(player_keys))
print("Random seeds:", RANDOM_SEEDS)
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# --------------------------------------------------
# 2. Define the preprocessing and model pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

def make_model():
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

# --------------------------------------------------
# 3. Repeat the player-identity split and evaluation
# --------------------------------------------------

results = []

for seed in RANDOM_SEEDS:
    train_players, test_players = train_test_split(
        player_keys,
        test_size=0.20,
        random_state=seed,
    )

    train_players = set(train_players)
    test_players = set(test_players)

    p1_train = data["player1_key"].isin(train_players)
    p2_train = data["player2_key"].isin(train_players)

    p1_test = data["player1_key"].isin(test_players)
    p2_test = data["player2_key"].isin(test_players)

    train_data = data[p1_train & p2_train].copy()
    test_data = data[p1_test & p2_test].copy()

    train_used = (
        set(train_data["player1_key"])
        | set(train_data["player2_key"])
    )
    test_used = (
        set(test_data["player1_key"])
        | set(test_data["player2_key"])
    )

    overlap = train_used & test_used

    if overlap:
        raise ValueError(
            f"Player leakage detected for seed {seed}!"
        )

    if train_data.empty or test_data.empty:
        print(f"Seed {seed}: skipped because a split is empty.")
        continue

    X_train = train_data[
        numeric_features + categorical_features
    ]
    X_test = test_data[
        numeric_features + categorical_features
    ]

    y_train = train_data["player1_win"]
    y_test = test_data["player1_win"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Seed {seed}: skipped because a split has one class.")
        continue

    model = make_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append(
        {
            "seed": seed,
            "training_matches": len(train_data),
            "test_matches": len(test_data),
            "training_players_used": len(train_used),
            "test_players_used": len(test_used),
            "player_overlap": len(overlap),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": auc,
        }
    )

    print(
        f"Seed {seed}: "
        f"train={len(train_data)}, "
        f"test={len(test_data)}, "
        f"accuracy={accuracy:.4f}, "
        f"balanced_accuracy={balanced_accuracy:.4f}, "
        f"ROC_AUC={auc:.4f}, "
        f"overlap={len(overlap)}"
    )

# --------------------------------------------------
# 4. Save and summarize results
# --------------------------------------------------

if not results:
    raise ValueError("No successful evaluation runs.")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
]

print("\n=== SUMMARY ACROSS SUCCESSFUL RUNS ===")
print("Successful runs:", len(results_df))
print("Results saved to:", RESULTS_PATH)

for metric in metric_columns:
    mean_value = results_df[metric].mean()
    std_value = results_df[metric].std(ddof=1)

    print(
        f"{metric}: {mean_value:.4f} "
        f"+/- {std_value:.4f} (standard deviation)"
    )

print("\n=== ALL RUN RESULTS ===")
print(results_df.to_string(index=False))

print("\nStability evaluation finished.")
