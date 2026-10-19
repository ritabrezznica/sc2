import pandas as pd

paired = pd.read_csv("sc2_paired_training.csv")
labels = pd.read_csv("replay_winner_labels.csv")

print("=== LABEL FILE ===")
print("Rows:", len(labels))
print("Unique replays:", labels["replay_index"].nunique())
print("Rows per result:")
print(labels["result"].value_counts(dropna=False).to_string())
print("Rows per winner_label:")
print(labels["winner_label"].value_counts(dropna=False).to_string())

print("\n=== CHECK WINNER LABELS PER REPLAY ===")
per_replay = labels.groupby("replay_index").agg(
    rows=("player_id", "size"),
    unique_players=("player_id", "nunique"),
    wins=("winner_label", "sum"),
    unique_results=("result", "nunique")
)
print("Replays with other than 2 rows:", int((per_replay["rows"] != 2).sum()))
print("Replays with other than 2 player IDs:", int((per_replay["unique_players"] != 2).sum()))
print("Replays without exactly one winner:", int((per_replay["wins"] != 1).sum()))
print("Replays with different result values:", int((per_replay["unique_results"] != 2).sum()))

print("\n=== SAMPLE WINNER MAPPINGS ===")
print(labels.head(12).to_string(index=False))

print("\n=== PAIRED DATASET LABEL COUNTS ===")
print(paired["player1_win"].value_counts().sort_index().to_string())

print("\n=== PAIRED REPLAY MATCH ===")
print("Paired unique replays:", paired["replay_index"].nunique())
print("Label unique replays:", labels["replay_index"].nunique())
print("Replay IDs identical:",
      set(paired["replay_index"]) == set(labels["replay_index"]))

print("\nDiagnostic complete.")
