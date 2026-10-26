
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
OUTPUT_PATH = "unseen_player_baseline_comparison.csv"

SEEDS = [11, 21, 31, 41, 42, 51, 61, 71, 81, 91]

# --------------------------------------------------
# 1. Load validated data
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

data = data.dropna(
    subset=[
        "player1_key",
        "player2_key",
        "player1_race",
        "player2_race",
    ]
).copy()

for col in ["player1_key", "player2_key"]:
    data[col] = data[col].astype(str).str.strip()

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

excluded = {
    "replay_index",
    "player1_win",
    "player1_key",
    "player2_key",
}

numeric_features = [
    c for c in training.columns if c not in excluded
]
categorical_features = ["player1_race", "player2_race"]

# --------------------------------------------------
# 2. Model pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

def make_model():
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

# --------------------------------------------------
# 3. Evaluate each split and its baselines
# --------------------------------------------------

rows = []

for seed in SEEDS:
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

    if train_data.empty or test_data.empty:
        print(f"Seed {seed}: skipped; empty split.")
        continue

    train_used = (
        set(train_data["player1_key"])
        | set(train_data["player2_key"])
    )
    test_used = (
        set(test_data["player1_key"])
        | set(test_data["player2_key"])
    )

    if train_used & test_used:
        raise ValueError(f"Identity overlap for seed {seed}")

    X_train = train_data[numeric_features + categorical_features]
    X_test = test_data[numeric_features + categorical_features]
    y_train = train_data["player1_win"]
    y_test = test_data["player1_win"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Seed {seed}: skipped; only one class.")
        continue

    model = make_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    # Baseline: always predict the majority class in the training set.
    majority_class = int(y_train.value_counts().idxmax())
    majority_pred = np.full(len(y_test), majority_class)

    # Reference baseline: always predict Player 1 wins.
    player1_win_pred = np.ones(len(y_test), dtype=int)

    row = {
        "seed": seed,
        "training_matches": len(train_data),
        "test_matches": len(test_data),
        "test_player1_win_rate": y_test.mean(),
        "model_accuracy": accuracy_score(y_test, pred),
        "model_balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "model_roc_auc": roc_auc_score(y_test, prob),
        "majority_class": majority_class,
        "majority_baseline_accuracy": accuracy_score(
            y_test, majority_pred
        ),
        "majority_baseline_balanced_accuracy": balanced_accuracy_score(
            y_test, majority_pred
        ),
        "always_player1_win_accuracy": accuracy_score(
            y_test, player1_win_pred
        ),
        "always_player1_win_balanced_accuracy": balanced_accuracy_score(
            y_test, player1_win_pred
        ),
        "player_overlap": len(train_used & test_used),
    }

    rows.append(row)

    print(
        f"Seed {seed}: "
        f"model accuracy={row['model_accuracy']:.4f}, "
        f"majority accuracy={row['majority_baseline_accuracy']:.4f}, "
        f"model balanced accuracy="
        f"{row['model_balanced_accuracy']:.4f}, "
        f"majority balanced accuracy="
        f"{row['majority_baseline_balanced_accuracy']:.4f}, "
        f"AUC={row['model_roc_auc']:.4f}"
    )

# --------------------------------------------------
# 4. Save results and summarize
# --------------------------------------------------

results = pd.DataFrame(rows)

if results.empty:
    raise ValueError("No successful evaluation runs.")

results.to_csv(OUTPUT_PATH, index=False)

print("\n=== MEAN RESULTS ACROSS SPLITS ===")
for col in [
    "model_accuracy",
    "majority_baseline_accuracy",
    "always_player1_win_accuracy",
    "model_balanced_accuracy",
    "majority_baseline_balanced_accuracy",
    "always_player1_win_balanced_accuracy",
    "model_roc_auc",
]:
    print(f"{col}: {results[col].mean():.4f}")

print("\nMean accuracy improvement over majority baseline:",
      f"{(results['model_accuracy'] - results['majority_baseline_accuracy']).mean():.4f}")

print("\nMean balanced-accuracy improvement over majority baseline:",
      f"{(results['model_balanced_accuracy'] - results['majority_baseline_balanced_accuracy']).mean():.4f}")

print("\nPlayer overlap across runs:",
      results["player_overlap"].tolist())

print("\nSaved:", OUTPUT_PATH)
