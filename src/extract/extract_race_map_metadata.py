import csv
import json
import os

CSV_PATH = "playerstats_at_60s_valid.csv"
JSON_PATH = r"sc2egset_merged\sc2egset_merged.json"
OFFSETS_PATH = r"sc2egset_merged\sc2egset_merged_offsets.json"

OUTPUT_PATH = "replay_race_map_metadata.csv"
ERROR_PATH = "replay_race_map_errors.csv"


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
                for p in toon_map.values():
                    if not isinstance(p, dict):
                        continue
                    try:
                        pid = int(p.get("playerID"))
                    except (ValueError, TypeError):
                        continue
                    if pid in (1, 2):
                        if pid in players:
                            raise ValueError(f"Duplicate playerID {pid}")
                        players[pid] = p

                if set(players) != {1, 2}:
                    raise ValueError(
                        f"Expected player IDs 1 and 2; found {sorted(players)}"
                    )

                p1, p2 = players[1], players[2]
                race1, race2 = p1.get("race"), p2.get("race")
                result1, result2 = p1.get("result"), p2.get("result")

                if not race1 or not race2:
                    raise ValueError("Missing player race")
                if result1 not in ("Win", "Loss") or result2 not in ("Win", "Loss"):
                    raise ValueError(
                        f"Invalid results: player1={result1}, player2={result2}"
                    )
                if result1 == result2:
                    raise ValueError(
                        f"Results are not one Win and one Loss: {result1}, {result2}"
                    )

                metadata = record.get("metadata") or {}
                map_name = metadata.get("mapName")
                if not map_name:
                    raise ValueError("Missing map name")

                records.append({
                    "replay_index": rid,
                    "player1_race": race1,
                    "player2_race": race2,
                    "map_name": map_name,
                    "player1_win_json": int(result1 == "Win"),
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

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "replay_index", "player1_race", "player2_race",
            "map_name", "player1_win_json"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    with open(ERROR_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["replay_index", "error", "details"]
        )
        writer.writeheader()
        writer.writerows(errors)

    print("\n=== RACE/MAP EXTRACTION SUMMARY ===")
    print("Input replay IDs:", len(valid_ids))
    print("Successfully extracted:", len(records))
    print("Errors/excluded replays:", len(errors))
    print("Metadata output:", OUTPUT_PATH)
    print("Error report:", ERROR_PATH)

    if records:
        print("\nPlayer 1 races:")
        counts = {}
        for r in records:
            counts[r["player1_race"]] = counts.get(r["player1_race"], 0) + 1
        for race, count in sorted(counts.items()):
            print(f"  {race}: {count}")

        print("\nPlayer 2 races:")
        counts = {}
        for r in records:
            counts[r["player2_race"]] = counts.get(r["player2_race"], 0) + 1
        for race, count in sorted(counts.items()):
            print(f"  {race}: {count}")


if __name__ == "__main__":
    main()
