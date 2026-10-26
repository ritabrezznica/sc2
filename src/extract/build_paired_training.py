import csv
from collections import defaultdict

STATS_FILE = "playerstats_at_60s_valid.csv"
LABELS_FILE = "replay_winner_labels.csv"
OUTPUT_FILE = "sc2_paired_training.csv"

def normalize_id(value):
    value = str(value).strip()
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value

# Read the 60-second statistics
with open(STATS_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    stats_fields = reader.fieldnames
    stats_rows = list(reader)

# Use scoreValue columns as model features
feature_fields = [
    col for col in stats_fields
    if col.startswith("scoreValue")
]

if not feature_fields:
    raise ValueError(
        "No columns beginning with 'scoreValue' were found. "
        f"Available columns: {stats_fields}"
    )

stats_by_replay = defaultdict(dict)

for row in stats_rows:
    replay = normalize_id(row["replay_index"])
    player = normalize_id(row["playerId"])

    if player in stats_by_replay[replay]:
        raise ValueError(
            f"Duplicate statistics row for replay {replay}, player {player}"
        )

    stats_by_replay[replay][player] = row

del stats_rows

# Read winner labels and group them by replay
labels_by_replay = defaultdict(list)

with open(LABELS_FILE, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        replay = normalize_id(row["replay_index"])
        player = normalize_id(row["player_id"])
        labels_by_replay[replay].append({
            "player": player,
            "winner_label": int(row["winner_label"])
        })

# Build one row per replay, preserving both opponents
output_rows = []
skipped = []

for replay, labels in labels_by_replay.items():
    if len(labels) != 2:
        skipped.append((replay, "not exactly 2 labels"))
        continue

    if {item["winner_label"] for item in labels} != {0, 1}:
        skipped.append((replay, "labels are not one Win and one Loss"))
        continue

    players = sorted(item["player"] for item in labels)

    if replay not in stats_by_replay:
        skipped.append((replay, "statistics missing"))
        continue

    replay_stats = stats_by_replay[replay]

    if len(replay_stats) != 2 or set(players) != set(replay_stats):
        skipped.append((replay, "player IDs do not match statistics"))
        continue

    label_by_player = {
        item["player"]: item["winner_label"] for item in labels
    }

    result = {"replay_index": replay}

    for number, player in enumerate(players, start=1):
        for feature in feature_fields:
            result[f"player{number}_{feature}"] = replay_stats[player][feature]

    # Target: did the first player (lowest player ID) win?
    result["player1_win"] = label_by_player[players[0]]

    output_rows.append(result)

if not output_rows:
    raise ValueError("No paired rows were created. Check the CSV column names.")

fieldnames = (
    ["replay_index"]
    + [f"player1_{x}" for x in feature_fields]
    + [f"player2_{x}" for x in feature_fields]
    + ["player1_win"]
)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print("Paired dataset created:", OUTPUT_FILE)
print("Paired replays:", len(output_rows))
print("Statistics used per player:", len(feature_fields))
print("Skipped replays:", len(skipped))
print("Target counts:", {
    "player1_win=1": sum(r["player1_win"] == 1 for r in output_rows),
    "player1_win=0": sum(r["player1_win"] == 0 for r in output_rows)
})
if skipped:
    print("First skipped examples:", skipped[:10])
