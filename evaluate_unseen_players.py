
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
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42

TRAINING_PATH = "sc2_paired_training.csv"
IDENTITY_PATH = "replay_player_identity_metadata.csv"

# --------------------------------------------------
# 1. Load and merge the validated datasets
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

# Player identities are required for this evaluation.
data = data.dropna(
    subset=[
        "player1_key",
        "player2_key",
        "player1_race",
        "player2_race",
    ]
).copy()

# Remove empty identity strings, if any.
for column in ["player1_key", "player2_key"]:
    data[column] = data[column].astype(str).str.strip()

data = data[
    (data["player1_key"] != "")
    & (data["player2_key"] != "")
    & (data["player1_key"].str.lower() != "nan")
    & (data["player2_key"].str.lower() != "nan")
].copy()

# --------------------------------------------------
# 2. Split PLAYER IDENTITIES, not individual matches
# --------------------------------------------------

player_keys = pd.unique(
    pd.concat(
        [data["player1_key"], data["player2_key"]],
        ignore_index=True,
    )
)

train_players, test_players = train_test_split(
    player_keys,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_players = set(train_players)
test_players = set(test_players)

player1_is_train = data["player1_key"].isin(train_players)
player2_is_train = data["player2_key"].isin(train_players)

player1_is_test = data["player1_key"].isin(test_players)
player2_is_test = data["player2_key"].isin(test_players)

train_data = data[
    player1_is_train & player2_is_train
].copy()

test_data = data[
    player1_is_test & player2_is_test
].copy()

cross_group_data = data[
    ~(
        (player1_is_train & player2_is_train)
        | (player1_is_test & player2_is_test)
    )
].copy()

print("=== UNSEEN-PLAYER SPLIT ===")
print("Matches with valid identities:", len(data))
print("Unique player identities:", len(player_keys))
print("Training-player identities:", len(train_players))
print("Unseen test-player identities:", len(test_players))
print("Training matches:", len(train_data))
print("Unseen-player test matches:", len(test_data))
print("Cross-group matches excluded:", len(cross_group_data))

if train_data.empty or test_data.empty:
    raise ValueError(
        "Training or test set is empty. "
        "Inspect the player split before continuing."
    )

# Confirm that no player identity occurs in both sets.
train_keys_used = set(train_data["player1_key"]) | set(
    train_data["player2_key"]
)
test_keys_used = set(test_data["player1_key"]) | set(
    test_data["player2_key"]
)

overlap = train_keys_used & test_keys_used

print("Player identities shared by train and test:", len(overlap))

if overlap:
    raise ValueError("Player leakage detected!")

# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------

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

categorical_features = [
    "player1_race",
    "player2_race",
]

X_train = train_data[
    numeric_features + categorical_features
].copy()

X_test = test_data[
    numeric_features + categorical_features
].copy()

y_train = train_data["player1_win"].astype(int)
y_test = test_data["player1_win"].astype(int)

print("\n=== FEATURE INFORMATION ===")
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())

if y_test.nunique() < 2:
    raise ValueError(
        "Test set contains only one target class; ROC AUC cannot be calculated."
    )

# --------------------------------------------------
# 4. Train a model using training players only
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Evaluate on completely unseen players
# --------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n=== UNSEEN-PLAYER RESULTS ===")
print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, predictions):.4f}"
)
print(f"ROC AUC:           {roc_auc_score(y_test, probabilities):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Player 1 loses", "Player 1 wins"],
        zero_division=0,
    )
)

print("\nUnseen-player evaluation finished.")
