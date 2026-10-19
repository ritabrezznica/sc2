
import pandas as pd

TRAINING_PATH = "sc2_paired_training.csv"
IDENTITY_PATH = "replay_player_identity_metadata.csv"
ERROR_PATH = "replay_player_identity_errors.csv"

training = pd.read_csv(TRAINING_PATH, encoding="utf-8-sig")
identity = pd.read_csv(IDENTITY_PATH, encoding="utf-8-sig")
errors = pd.read_csv(ERROR_PATH, encoding="utf-8-sig")

print("=== FILE SIZES ===")
print("Training matches:", len(training))
print("Identity records:", len(identity))
print("Error records:", len(errors))

print("\n=== DUPLICATE REPLAY IDS ===")
print("Training duplicates:", training["replay_index"].duplicated().sum())
print("Identity duplicates:", identity["replay_index"].duplicated().sum())

# Keep one row per replay only if there are no duplicate IDs.
if training["replay_index"].duplicated().any():
    raise ValueError("Duplicate replay IDs in training data.")

if identity["replay_index"].duplicated().any():
    raise ValueError("Duplicate replay IDs in identity metadata.")

# Match the identity records to the training dataset.
merged = training.merge(
    identity,
    on="replay_index",
    how="left",
    suffixes=("_training", "_identity"),
    indicator=True
)

missing = merged[merged["_merge"] == "left_only"]
matched = merged[merged["_merge"] == "both"]

print("\n=== MATCH COVERAGE ===")
print("Matched training replays:", len(matched))
print("Training replays without identity:", len(missing))

print("\n=== PLAYER KEY COVERAGE ===")
for col in ["player1_key", "player2_key"]:
    if col in merged.columns:
        print(f"{col} missing:", merged[col].isna().sum())
        print(f"{col} unique:", merged[col].nunique())

# Compare the winner label extracted from JSON with the training label.
if "player1_win_json" in merged.columns:
    valid = merged[
        merged["_merge"].eq("both")
        & merged["player1_win_json"].notna()
    ].copy()

    json_win = valid["player1_win_json"].astype(str).str.strip().str.lower()
    training_win = valid["player1_win"].astype(int).astype(str)

    # Handle common JSON representations of boolean/numeric labels.
    json_win = json_win.replace({
        "true": "1",
        "false": "0",
        "win": "1",
        "loss": "0",
        "1.0": "1",
        "0.0": "0",
    })

    mismatch = json_win != training_win

    print("\n=== WINNER LABEL VALIDATION ===")
    print("Labels compared:", len(valid))
    print("Label mismatches:", int(mismatch.sum()))

    if mismatch.any():
        print(valid.loc[
            mismatch,
            ["replay_index", "player1_win", "player1_win_json"]
        ].head(10).to_string(index=False))

print("\n=== ERROR REPORT SAMPLE ===")
print(errors.head(10).to_string(index=False))

print("\nValidation finished.")
