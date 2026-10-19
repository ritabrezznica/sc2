import csv
import json
import os

CSV_PATH = "playerstats_at_60s_valid.csv"
JSON_PATH = r"sc2egset_merged\sc2egset_merged.json"
OFFSETS_PATH = r"sc2egset_merged\sc2egset_merged_offsets.json"

OUTPUT_PATH = "replay_winner_labels.csv"
ERROR_PATH = "winner_label_errors.csv"


def load_valid_replay_ids(path):
    ids = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ids.add(int(row["replay_index"]))
    return sorted(ids)


def extract_players(toon_map):
    """Find player records with a Win/Loss result."""
    found = []

    def walk(obj, inherited_id=None):
        if isinstance(obj, dict):
            # A numeric dictionary key may identify the player.
            pid = inherited_id
            for key in ("playerId", "playerID", "player_id", "id"):
                if key in obj:
                    try:
                        pid = int(obj[key])
                        break
                    except (ValueError, TypeError):
                        pass

            result = obj.get("result")
            if isinstance(result, str):
                result = result.strip()

            if result in ("Win", "Loss"):
                name = None
                for key in ("name", "playerName", "toonName", "nickname"):
                    if obj.get(key):
                        name = str(obj[key])
                        break

                found.append({
                    "player_id": pid,
                    "player_name": name or "",
                    "result": result,
                })

            for key, value in obj.items():
                child_id = inherited_id
                try:
                    numeric_key = int(key)
                    if numeric_key in (1, 2):
                        child_id = numeric_key
                except (ValueError, TypeError):
                    pass
                if isinstance(value, (dict, list)):
                    walk(value, child_id)

        elif isinstance(obj, list):
            for value in obj:
                walk(value, inherited_id)

    walk(toon_map)

    # Remove exact duplicate records, if the structure repeats them.
    unique = {}
    for p in found:
        key = (p["player_id"], p["player_name"], p["result"])
        unique[key] = p

    return list(unique.values())


def main():
    valid_ids = load_valid_replay_ids(CSV_PATH)

    with open(OFFSETS_PATH, encoding="utf-8") as f:
        offsets = json.load(f)

    json_size = os.path.getsize(JSON_PATH)

    labels = []
    errors = []
    winner_count = 0
    processed = 0

    with open(JSON_PATH, "rb") as f:
        for n, rid in enumerate(valid_ids, start=1):
            if rid < 1 or rid > len(offsets):
                errors.append({
                    "replay_index": rid,
                    "error": "Replay index outside offsets range",
                    "details": "",
                })
                continue

            start = offsets[rid - 1]
            end = offsets[rid] if rid < len(offsets) else json_size

            try:
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8").lstrip()
                record, _ = json.JSONDecoder().raw_decode(chunk)

                toon_map = record.get("ToonPlayerDescMap")
                if not isinstance(toon_map, (dict, list)):
                    raise ValueError("ToonPlayerDescMap missing or unexpected type")

                players = extract_players(toon_map)

                # Require exactly two distinct players, with IDs 1 and 2.
                if len(players) != 2:
                    raise ValueError(
                        f"Expected 2 player result records; found {len(players)}"
                    )

                ids = [p["player_id"] for p in players]
                results = [p["result"] for p in players]

                if set(ids) != {1, 2}:
                    raise ValueError(f"Unexpected or missing player IDs: {ids}")

                if sorted(results) != ["Loss", "Win"]:
                    raise ValueError(f"Expected one Win and one Loss; found {results}")

                by_id = {p["player_id"]: p for p in players}
                processed += 1
                winner_count += 1

                for pid in (1, 2):
                    p = by_id[pid]
                    labels.append({
                        "replay_index": rid,
                        "player_id": pid,
                        "player_name": p["player_name"],
                        "result": p["result"],
                        "winner_label": 1 if p["result"] == "Win" else 0,
                    })

            except Exception as e:
                errors.append({
                    "replay_index": rid,
                    "error": type(e).__name__,
                    "details": str(e),
                })

            if n % 500 == 0 or n == len(valid_ids):
                print(f"Checked {n:,}/{len(valid_ids):,} replays; "
                      f"valid={processed:,}; errors={len(errors):,}")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "replay_index", "player_id", "player_name",
            "result", "winner_label"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(labels)

    with open(ERROR_PATH, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["replay_index", "error", "details"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    print("\n=== WINNER LABEL EXTRACTION SUMMARY ===")
    print("Valid CSV replay IDs:", len(valid_ids))
    print("Successfully processed replays:", processed)
    print("Replays with errors:", len(errors))
    print("Player label rows written:", len(labels))
    print("Output:", OUTPUT_PATH)
    print("Error report:", ERROR_PATH)

    if not errors and processed == len(valid_ids):
        print("RESULT: Every valid replay has verified winner labels.")
    else:
        print("RESULT: Review the error report before proceeding.")


if __name__ == "__main__":
    main()
