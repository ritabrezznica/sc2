import os
import json

DATA_DIR = "./data/unpacked"
JSON_FILE = None

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.SC2Replay.json'):
            JSON_FILE = os.path.join(root, file)
            break

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    match_data = json.load(f)

print("--> Top-level keys in JSON:")
print(list(match_data.keys()))

# Check sample keys if nested
for key in list(match_data.keys())[:5]:
    val = match_data[key]
    if isinstance(val, list) and len(val) > 0:
        print(f"\n--> First entry in list key '{key}':")
        print(val[0])
    elif isinstance(val, dict):
        print(f"\n--> Keys inside dict key '{key}':")
        print(list(val.keys()))