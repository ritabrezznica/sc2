import csv
import sys
import ijson

INPUT = r"sc2egset_merged\sc2egset_merged.json"
RAW_OUTPUT = r"playerstats_raw.csv"
TARGET_OUTPUT = r"playerstats_at_60s.csv"

# Provisional target: 960 game loops.
# We will verify the loop-to-time conversion separately.
TARGET_LOOP = 960

STAT_KEYS = [
    "scoreValueFoodMade",
    "scoreValueFoodUsed",
    "scoreValueMineralsCollectionRate",
    "scoreValueMineralsCurrent",
    "scoreValueMineralsFriendlyFireArmy",
    "scoreValueMineralsFriendlyFireEconomy",
    "scoreValueMineralsFriendlyFireTechnology",
    "scoreValueMineralsKilledArmy",
    "scoreValueMineralsKilledEconomy",
    "scoreValueMineralsKilledTechnology",
    "scoreValueMineralsLostArmy",
    "scoreValueMineralsLostEconomy",
    "scoreValueMineralsLostTechnology",
    "scoreValueMineralsUsedActiveForces",
    "scoreValueMineralsUsedCurrentArmy",
    "scoreValueMineralsUsedCurrentEconomy",
    "scoreValueMineralsUsedCurrentTechnology",
    "scoreValueMineralsUsedInProgressArmy",
    "scoreValueMineralsUsedInProgressEconomy",
    "scoreValueMineralsUsedInProgressTechnology",
    "scoreValueVespeneCollectionRate",
    "scoreValueVespeneCurrent",
    "scoreValueVespeneFriendlyFireArmy",
    "scoreValueVespeneFriendlyFireEconomy",
    "scoreValueVespeneFriendlyFireTechnology",
    "scoreValueVespeneKilledArmy",
    "scoreValueVespeneKilledEconomy",
    "scoreValueVespeneKilledTechnology",
    "scoreValueVespeneLostArmy",
    "scoreValueVespeneLostEconomy",
    "scoreValueVespeneLostTechnology",
    "scoreValueVespeneUsedActiveForces",
    "scoreValueVespeneUsedCurrentArmy",
    "scoreValueVespeneUsedCurrentEconomy",
    "scoreValueVespeneUsedCurrentTechnology",
    "scoreValueVespeneUsedInProgressArmy",
    "scoreValueVespeneUsedInProgressEconomy",
    "scoreValueVespeneUsedInProgressTechnology",
    "scoreValueWorkersActiveCount",
]

BASE_FIELDS = ["replay_index", "loop", "playerId"]
RAW_FIELDS = BASE_FIELDS + STAT_KEYS
TARGET_FIELDS = RAW_FIELDS + ["target_loop", "loop_distance"]

raw_count = 0
target_count = 0
replay_count = 0

with open(INPUT, "rb") as source, \
     open(RAW_OUTPUT, "w", newline="", encoding="utf-8-sig") as raw_file, \
     open(TARGET_OUTPUT, "w", newline="", encoding="utf-8-sig") as target_file:

    raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
    target_writer = csv.DictWriter(target_file, fieldnames=TARGET_FIELDS)

    raw_writer.writeheader()
    target_writer.writeheader()

    # The input is a top-level JSON array of replay objects.
    for replay_index, replay in enumerate(ijson.items(source, "item"), 1):
        replay_count = replay_index
        closest = {}

        for event in replay.get("trackerEvents", []):
            if event.get("evtTypeName") != "PlayerStats":
                continue

            loop = event.get("loop")
            player_id = event.get("playerId")

            if loop is None or player_id is None:
                continue

            stats = event.get("stats") or {}

            row = {
                "replay_index": replay_index,
                "loop": loop,
                "playerId": player_id,
            }

            for key in STAT_KEYS:
                row[key] = stats.get(key, "")

            raw_writer.writerow(row)
            raw_count += 1

            # Keep the closest snapshot to TARGET_LOOP for each player.
            key = player_id
            distance = abs(loop - TARGET_LOOP)

            if key not in closest or distance < closest[key][0]:
                closest[key] = (distance, row.copy())

        for distance, row in closest.values():
            target_row = row.copy()
            target_row["target_loop"] = TARGET_LOOP
            target_row["loop_distance"] = distance
            target_writer.writerow(target_row)
            target_count += 1

        if replay_index % 100 == 0:
            print(
                f"Replays: {replay_index:,} | "
                f"PlayerStats rows: {raw_count:,} | "
                f"Selected snapshots: {target_count:,}",
                file=sys.stderr,
                flush=True,
            )

print(f"Finished. Replays processed: {replay_count:,}")
print(f"Raw PlayerStats rows: {raw_count:,}")
print(f"Selected snapshots: {target_count:,}")
print(f"Created: {RAW_OUTPUT}")
print(f"Created: {TARGET_OUTPUT}")
