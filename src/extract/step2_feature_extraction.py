import os
import json
import pandas as pd

DATA_DIR = "./data/unpacked"
JSON_FILE = None

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.SC2Replay.json'):
            JSON_FILE = os.path.join(root, file)
            break

if not JSON_FILE:
    raise FileNotFoundError("Could not locate .SC2Replay.json match log!")

print(f"--> Extracting Dual-Frame telemetry from: {os.path.basename(JSON_FILE)}")

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    match_data = json.load(f)

tracker_events = match_data.get('trackerEvents', [])
game_events = match_data.get('gameEvents', [])

TIME_WINDOWS = {'t_3min': 180, 't_4min': 240, 't_5min': 300}
WORKER_NAMES = {'drone', 'scv', 'probe'}

extracted_records = []

for label, target_sec in TIME_WINDOWS.items():
    target_loop = target_sec * 22.4
    
    window_tracker = [e for e in tracker_events if e.get('loop', 0) <= target_loop]
    window_game = [e for e in game_events if e.get('loop', 0) <= target_loop]
    
    p1_workers, p2_workers = 0, 0
    p1_buildings, p2_buildings = 0, 0
    
    for e in window_tracker:
        evt_type = str(e.get('evtTypeName', ''))
        unit_name = str(e.get('unitTypeName', '') or e.get('unitType', '')).lower()
        # Accept both 1-based (playerId) and 0-based (controlPlayerId/upkeepPlayerId) schemas
        p_id = e.get('playerId', e.get('controlPlayerId', e.get('upkeepPlayerId', -1)))
        
        if evt_type in ['UnitBorn', 'UnitInit', 'UnitDone']:
            if any(w in unit_name for w in WORKER_NAMES):
                if p_id in [1, 0]: p1_workers += 1
                elif p_id in [2, 1]: p2_workers += 1
            elif evt_type in ['UnitInit', 'BuildingInit']:
                if p_id in [1, 0]: p1_buildings += 1
                elif p_id in [2, 1]: p2_buildings += 1

    # Extract user commands by user ID
    p1_actions = len([e for e in window_game if e.get('userid', {}).get('userId') == 0 or e.get('userId') == 0])
    p2_actions = len([e for e in window_game if e.get('userid', {}).get('userId') == 1 or e.get('userId') == 1])
    
    p1_apm = round(p1_actions / (target_sec / 60), 2)
    p2_apm = round(p2_actions / (target_sec / 60), 2)
    
    extracted_records.append({
        'time_window': label,
        'checkpoint_seconds': target_sec,
        'p1_apm': p1_apm,
        'p1_workers': p1_workers,
        'p1_buildings': p1_buildings,
        'delta_apm': round(p1_apm - p2_apm, 2),
        'delta_workers': p1_workers - p2_workers,
        'delta_buildings': p1_buildings - p2_buildings,
    })

df_features = pd.DataFrame(extracted_records)
os.makedirs("./data/processed", exist_ok=True)
df_features.to_csv("./data/processed/dual_frame_features.csv", index=False)

print("\n================ REFIXED EXTRACTED DUAL-FRAME MATRIX ================")
print(df_features.to_string(index=False))
print("====================================================================\n")