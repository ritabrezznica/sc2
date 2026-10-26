
import csv
import json
import os
from collections import Counter

CSV_PATH = "playerstats_at_60s_valid.csv"
JSON_PATH = r"sc2egset_merged\sc2egset_merged.json"
OFFSETS_PATH = r"sc2egset_merged\sc2egset_merged_offsets.json"

OUTPUT_PATH = "replay_player_identity_metadata.csv"
ERROR_PATH = "replay_player_identity_errors.csv"


def load_valid_replay_ids(path):
    ids = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ids.add(int(row["replay_index"]))
    return sorted(ids)


def main():
    valid_ids = load_valid_replay_ids(CSV_PATH)

    with open(OFFSETS_PATH, encoding="utf-8") as f:
        offsets = json.load(f)

    json_size = os.path.getsize(JSON_PATH)
    records = []
    errors = []
    processed = 0

    with open(JSON_PATH, "rb") as f:
        for n, rid in enumerate(valid_ids, start=1):
            try:
                if rid < 1 or rid > len(offsets):
                    raise ValueError("Replay index outside offsets range")

                start = offsets[rid - 1]
                end = offsets[rid] if rid < len(offsets) else json_size

                f.seek(start)
                chunk = f.read(end - start).decode("utf-8").lstrip()
                record, _ = json.JSONDecoder().raw_decode(chunk)

                toon_map = record.get("ToonPlayerDescMap")
                if not isinstance(toon_map, dict):
                    raise ValueError("ToonPlayerDescMap is not a dictionary")

                players = {}

                for player_key, p in toon_map.items():
                    if not isinstance(p, dict):
                        continue

                    try:
                        pid = int(p.get("playerID"))
                    except (ValueError, TypeError):
                        continue

                    if pid not in (1, 2):
                        continue

                    if pid in players:
                        raise ValueError(f"Duplicate playerID {pid}")

                    players[pid] = {
                        "player_key": str(player_key),
                        "nickname": str(p.get("nickname") or ""),
                        "race": str(p.get("race") or ""),
                        "result": p.get("result"),
                    }

                if set(players) != {1, 2}:
                    raise ValueError(
                        f"Expected players 1 and 2; found {sorted(players)}"
                    )

                p1, p2 = players[1], players[2]

                if not p1["player_key"] or not p2["player_key"]:
                    raise ValueError("Missing player identifier")

                if p1["player_key"] == p2["player_key"]:
                    raise ValueError("Both players have the same identifier")

                if p1["result"] not in ("Win", "Loss"):
                    raise ValueError("Invalid player 1 result")

                if p2["result"] not in ("Win", "Loss"):
                    raise ValueError("Invalid player 2 result")

                if p1["result"] == p2["result"]:
                    raise ValueError("Expected one Win and one Loss")

                records.append({
                    "replay_index": rid,
                    "player1_key": p1["player_key"],
                    "player2_key": p2["player_key"],
                    "player1_nickname": p1["nickname"],
                    "player2_nickname": p2["nickname"],
                    "player1_race": p1["race"],
                    "player2_race": p2["race"],
                    "player1_win_json": int(p1["result"] == "Win"),
                })

                processed += 1

            except Exception as e:
                errors.append({
                    "replay_index": rid,
                    "error": type(e).__name__,
                    "details": str(e),
                })

            if n % 500 == 0 or n == len(valid_ids):
                print(
                    f"Checked {n:,}/{len(valid_ids):,}; "
                    f"valid={processed:,}; errors={len(errors):,}"
                )

    fields = [
        "replay_index",
        "player1_key",
        "player2_key",
        "player1_nickname",
        "player2_nickname",
        "player1_race",
        "player2_race",
        "player1_win_json",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    with open(ERROR_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["replay_index", "error", "details"],
        )
        writer.writeheader()
        writer.writerows(errors)

    player_keys = Counter()
    nickname_by_key = {}

    for row in records:
        for side in ("player1", "player2"):
            key = row[f"{side}_key"]
            nickname = row[f"{side}_nickname"]

            player_keys[key] += 1
            nickname_by_key.setdefault(key, set()).add(nickname)

    repeated_keys = sum(
        count > 1 for count in player_keys.values()
    )
    keys_with_multiple_names = sum(
        len(names) > 1 for names in nickname_by_key.values()
    )

    print("\n=== PLAYER IDENTITY EXTRACTION SUMMARY ===")
    print("Input replay IDs:", len(valid_ids))
    print("Successfully extracted:", len(records))
    print("Errors/excluded replays:", len(errors))
    print("Unique player keys:", len(player_keys))
    print("Keys appearing in multiple matches:", repeated_keys)
    print("Keys with multiple displayed nicknames:", keys_with_multiple_names)
    print("Metadata output:", OUTPUT_PATH)
    print("Error report:", ERROR_PATH)


if __name__ == "__main__":
    main()
