import os
import json
import pandas as pd
import numpy as np

# Path to your SC2EGSet raw JSON files directory
REPLAY_DIR = "./replays_json"  # Update this to your local directory containing SC2EGSet JSON files
OUTPUT_CSV = "./data/processed/real_sc2egset_10k_matches.csv"

# Unit Classification Schemas
WORKER_TYPES = {"SCV", "Drone", "Probe"}
COMMAND_EVENT_TYPES = {"NHDT.CmdEvent", "CmdEvent", "Cmd"}

def parse_replay_json(filepath):
    """
    Parses a single SC2EGSet replay JSON log, extracts features strictly at t <= 300s,
    and returns the true outcome (Win/Loss for P1).
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        tracker_events = data.get("trackerEvents", [])
        game_events = data.get("gameEvents", [])
        
        # 1. Determine Match Outcome (True Y)
        # SC2EGSet stores winner in metadata or final unit/structure ownership
        p1_won = None
        if "winner" in data:
            p1_won = 1 if data["winner"] == 1 else 0
        else:
            # Fallback: Check structural survival/gg state if metadata absent
            p1_won = 1 if data.get("p1_result") == "Win" else 0
            
        if p1_won is None:
            return None # Skip replays with ambiguous outcomes

        # Initialize State Containers at t = 300s (6720 engine loops)
        max_loop = 300 * 22.4 # t = 300 seconds
        
        p1_cmd_count, p2_cmd_count = 0, 0
        p1_workers, p2_workers = 0, 0
        p1_buildings, p2_buildings = 0, 0
        p1_army_supply, p2_army_supply = 0, 0
        p1_unspent, p2_unspent = 0, 0

        # Parse Historical Command Actions (gameEvents) for 0 <= t <= 300s
        for evt in game_events:
            loop = evt.get("loop", 0)
            if loop > max_loop:
                break
            
            # Action Rate (Cmd Events)
            evt_type = evt.get("evtTypeName", "")
            if any(cmd in evt_type for cmd in COMMAND_EVENT_TYPES):
                player_id = evt.get("userId", evt.get("playerId", 0))
                if player_id == 0 or player_id == 1:
                    p1_cmd_count += 1
                elif player_id == 2:
                    p2_cmd_count += 1

        # Calculate Command APM (actions / 5 elapsed minutes)
        p1_apm = round(p1_cmd_count / 5.0, 2)
        p2_apm = round(p2_cmd_count / 5.0, 2)

        # Parse Point-in-Time State Engine Events (trackerEvents) for t <= 300s
        for evt in tracker_events:
            loop = evt.get("loop", 0)
            if loop > max_loop:
                break
            
            evt_type = evt.get("evtTypeName", "")
            
            # Economic & Military Unit Spawns
            if evt_type == "UnitBornEvent" or evt_type == "UnitInitEvent":
                unit_name = evt.get("unitTypeName", "")
                control_player = evt.get("controlPlayerId", evt.get("upkeepPlayerId", 0))
                
                # Workers
                if unit_name in WORKER_TYPES:
                    if control_player == 1: p1_workers += 1
                    elif control_player == 2: p2_workers += 1
                # Buildings
                elif "Building" in evt_type or evt.get("isBuilding", False):
                    if control_player == 1: p1_buildings += 1
                    elif control_player == 2: p2_buildings += 1
                # Military Army Supply
                else:
                    food_made = evt.get("foodMade", 0)
                    if food_made > 0:
                        if control_player == 1: p1_army_supply += food_made
                        elif control_player == 2: p2_army_supply += food_made

            # State Snapshots of Unspent Resources at or near t = 300s
            if evt_type == "PlayerStatsEvent":
                player_id = evt.get("playerId", 0)
                minerals = evt.get("stats", {}).get("mineralsCurrent", 0)
                vespene = evt.get("stats", {}).get("vespeneCurrent", 0)
                total_unspent = minerals + vespene
                
                if player_id == 1: p1_unspent = total_unspent
                elif player_id == 2: p2_unspent = total_unspent

        # Construct Real Differential Metrics
        delta_apm = p1_apm - p2_apm
        delta_workers = p1_workers - p2_workers
        delta_buildings = p1_buildings - p2_buildings
        delta_army = p1_army_supply - p2_army_supply
        delta_unspent = p1_unspent - p2_unspent

        return {
            'match_id': os.path.basename(filepath),
            'p1_apm': p1_apm, 'p1_workers': p1_workers, 'p1_buildings': p1_buildings, 
            'p1_army': p1_army_supply, 'p1_unspent': p1_unspent,
            'delta_apm': delta_apm, 'delta_workers': delta_workers, 
            'delta_buildings': delta_buildings, 'delta_army': delta_army, 
            'delta_unspent': delta_unspent,
            'win': p1_won
        }
        
    except Exception as e:
        # Gracefully handle corrupted JSON replays or patch schema variations
        return None

def build_dataset(replay_dir, max_matches=10000):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    all_files = [os.path.join(replay_dir, f) for f in os.listdir(replay_dir) if f.endswith('.json')]
    
    print(f"--> Found {len(all_files)} raw JSON replays. Extracting up to {max_matches} valid matches...")
    
    records = []
    for idx, filepath in enumerate(all_files):
        if len(records) >= max_matches:
            break
            
        record = parse_replay_json(filepath)
        if record is not None:
            records.append(record)
            
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(all_files)} files... Valid matches extracted: {len(records)}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Successfully parsed and saved {len(df)} real SC2EGSet match vectors to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Ensure REPLAY_DIR points to your folder containing the .SC2Replay.json files
    if os.path.exists(REPLAY_DIR):
        build_dataset(REPLAY_DIR, max_matches=10000)
    else:
        print(f"[ERROR] Replay directory '{REPLAY_DIR}' not found. Please set REPLAY_DIR to your local JSON replay path.")