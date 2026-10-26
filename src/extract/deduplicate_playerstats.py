import csv

input_file = "playerstats_raw.csv"
output_file = "playerstats_deduplicated.csv"

seen = set()
total = 0
removed = 0

with open(input_file, newline="", encoding="utf-8-sig") as src, \
     open(output_file, "w", newline="", encoding="utf-8-sig") as dst:

    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        total += 1
        signature = tuple(row.items())

        if signature in seen:
            removed += 1
            continue

        seen.add(signature)
        writer.writerow(row)

print("Original data rows:", total)
print("Exact duplicate rows removed:", removed)
print("Rows retained:", total - removed)
print("Output:", output_file)
