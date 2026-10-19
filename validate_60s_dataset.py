import csv
from collections import Counter, defaultdict

filename = "playerstats_at_60s.csv"
pair_counts = Counter()
players_by_replay = defaultdict(set)
missing_values = Counter()
loop_counts = Counter()
total_rows = 0

with open(filename, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total_rows += 1
        replay = row["replay_index"]
        player = row["playerId"]
        pair_counts[(replay, player)] += 1
        players_by_replay[replay].add(player)
        loop_counts[row["loop"]] += 1

        for column, value in row.items():
            if value is None or value.strip() == "":
                missing_values[column] += 1

duplicates = {
    key: count for key, count in pair_counts.items()
    if count > 1
}
wrong_players = {
    replay: sorted(players)
    for replay, players in players_by_replay.items()
    if players != {"1", "2"}
}

print("Total rows:", total_rows)
print("Unique replays:", len(players_by_replay))
print("Expected rows if 2 players per replay:",
      len(players_by_replay) * 2)
print("Repeated replay/player pairs:", len(duplicates))
print("Replays not containing exactly players 1 and 2:",
      len(wrong_players))
print("Rows with missing values:", sum(missing_values.values()))
print("Loop counts:", dict(loop_counts))

if duplicates:
    print("\nFirst 10 repeated pairs:")
    for key, count in list(duplicates.items())[:10]:
        print(f"Replay={key[0]}, Player={key[1]}, Rows={count}")

if wrong_players:
    print("\nFirst 10 replays with unexpected players:")
    for replay, players in list(wrong_players.items())[:10]:
        print(f"Replay={replay}, Players={players}")

if missing_values:
    print("\nColumns with missing values:")
    for column, count in missing_values.items():
        print(f"{column}: {count}")
