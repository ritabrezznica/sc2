import csv
from collections import defaultdict

source = "playerstats_raw.csv"
cleaned = "playerstats_cleaned.csv"
conflicts_file = "playerstats_conflicts.csv"

seen_rows = set()
cleaned_rows = []
groups = defaultdict(list)

with open(source, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = [name.strip() for name in reader.fieldnames]

    for raw_row in reader:
        row = {
            k.strip(): v
            for k, v in raw_row.items()
            if k is not None
        }

        key = (row["replay_index"], row["playerId"], row["loop"])
        row_tuple = tuple(row.get(name, "") for name in fieldnames)

        groups[key].append(row_tuple)

        # Remove only exact duplicate rows.
        if row_tuple not in seen_rows:
            seen_rows.add(row_tuple)
            cleaned_rows.append(row)

with open(cleaned, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned_rows)

conflict_keys = {
    key for key, rows in groups.items()
    if len(set(rows)) > 1
}

with open(conflicts_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in cleaned_rows:
        key = (row["replay_index"], row["playerId"], row["loop"])
        if key in conflict_keys:
            writer.writerow(row)

print("Original file:", source)
print("Original rows:", sum(len(rows) for rows in groups.values()))
print("Cleaned rows:", len(cleaned_rows))
print("Exact duplicate rows removed:",
      sum(len(rows) for rows in groups.values()) - len(cleaned_rows))
print("Conflicting replay/player/loop combinations:", len(conflict_keys))
print("Cleaned file:", cleaned)
print("Conflict report:", conflicts_file)
