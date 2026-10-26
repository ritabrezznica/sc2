import csv
from collections import Counter

path = "playerstats_at_60s.csv"

loops = Counter()
distances = Counter()
players_per_replay = Counter()
seen = set()
duplicates = 0
rows = 0

with open(path, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rows += 1
        replay = row["replay_index"]
        player = row["playerId"]
        loops[row["loop"]] += 1
        distances[row["loop_distance"]] += 1
        players_per_replay[replay] += 1

        key = (replay, player)
        if key in seen:
            duplicates += 1
        seen.add(key)

print("Total snapshot rows:", rows)
print("Unique replays:", len(players_per_replay))
print("Game-loop distribution:", loops.most_common(10))
print("Loop-distance distribution:", distances.most_common(10))
print("Players per replay:", Counter(players_per_replay.values()))
print("Duplicate replay/player pairs:", duplicates)
