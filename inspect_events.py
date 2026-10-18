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

tracker = match_data.get('trackerEvents', [])
game = match_data.get('gameEvents', [])

print("=== TRACKER EVENTS SAMPLE (FIRST 3) ===")
for e in tracker[:3]:
    print(e)

print("\n=== GAME EVENTS SAMPLE (FIRST 3) ===")
for e in game[:3]:
    print(e)

# Collect unique event names or types
tracker_types = set(e.get('_event', e.get('event', 'N/A')) for e in tracker)
game_types = set(e.get('_event', e.get('event', 'N/A')) for e in game)

print("\n=== UNIQUE TRACKER EVENT TYPES ===")
print(list(tracker_types)[:10])

print("\n=== UNIQUE GAME EVENT TYPES ===")
print(list(game_types)[:10])

# Inspect loop range
loops = [e.get('_loop', e.get('loop', 0)) for e in game + tracker]
if loops:
    print(f"\nLoop Range: Min = {min(loops)}, Max = {max(loops)}")