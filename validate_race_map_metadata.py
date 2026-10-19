import csv
from collections import Counter

METADATA_PATH = "replay_race_map_metadata.csv"
TRAINING_PATH = "sc2_paired_training.csv"

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

metadata = load_csv(METADATA_PATH)
training = load_csv(TRAINING_PATH)

metadata_by_id = {int(r["replay_index"]): r for r in metadata}
training_by_id = {int(r["replay_index"]): r for r in training}

metadata_ids = set(metadata_by_id)
training_ids = set(training_by_id)
common_ids = metadata_ids & training_ids

winner_mismatches = [
    rid for rid in common_ids
    if int(metadata_by_id[rid]["player1_win_json"])
    != int(training_by_id[rid]["player1_win"])
]

missing_metadata = training_ids - metadata_ids
metadata_only = metadata_ids - training_ids

matchups = Counter(
    (
        metadata_by_id[rid]["player1_race"],
        metadata_by_id[rid]["player2_race"],
    )
    for rid in common_ids
)

maps = Counter(metadata_by_id[rid]["map_name"] for rid in common_ids)

print("=== METADATA VALIDATION ===")
print("Metadata rows:", len(metadata))
print("Training rows:", len(training))
print("Matching replay IDs:", len(common_ids))
print("Training replays missing metadata:", len(missing_metadata))
print("Metadata replays absent from training:", len(metadata_only))
print("Winner-label mismatches:", len(winner_mismatches))

if winner_mismatches:
    print("First mismatches:", sorted(winner_mismatches)[:20])

print("\n=== RACE MATCHUPS ===")
for (race1, race2), count in sorted(matchups.items()):
    print(f"{race1} vs {race2}: {count}")

print("\n=== MAP SUMMARY ===")
print("Unique maps:", len(maps))
print("Top 10 maps:")
for map_name, count in maps.most_common(10):
    print(f"{map_name}: {count}")

if (
    len(common_ids) == len(training)
    and not missing_metadata
    and not metadata_only
    and not winner_mismatches
):
    print("\nRESULT: PASS - metadata and winner labels match training data.")
else:
    print("\nRESULT: REVIEW - check the differences reported above.")
