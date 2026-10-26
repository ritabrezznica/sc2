"""
Extract raw per-player features at 60s from replay JSONs.

This is a robust, best-effort extractor that:
- looks for JSON files under data/raw/ or data/replays/ (common folders)
- tries common structures: top-level 'players', 'frames', per-player timeseries, or simple keys
- extracts (if available) per-player minerals, vespene, supply/cap, and unit counts at ~60s
- falls back to the existing data/processed/all_match_features.csv when no raw data found
- writes data/processed/all_match_features_raw.csv with appended raw fields

You will likely need to adapt the parsing for your exact replay JSON schema; the script prints parsing diagnostics.
"""
import os
import glob
import json
import math
import pandas as pd

RAW_GLOBS = ["data/raw/*.json", "data/replays/*.json", "data/raw_replays/*.json"]
OUT_CSV = "data/processed/all_match_features_raw.csv"
BASE_CSV = "data/processed/all_match_features.csv"
SEC = 60
TOL = 2.0  # seconds tolerance when matching timestamps

def find_files():
    files = []
    for g in RAW_GLOBS:
        files.extend(glob.glob(g))
    return sorted(set(files))

def pick_value_at_time(times, values, target=SEC, tol=TOL):
    # times: list of timestamps (seconds) or indices; values: same length list
    if times is None or values is None:
        return None
    if len(times) != len(values):
        # try if values is dict keyed by time
        if isinstance(values, dict):
            # pick closest key
            try:
                keys = [float(k) for k in values.keys()]
                best = min(keys, key=lambda k: abs(k - target))
                if abs(best - target) <= tol:
                    return values[str(int(best))] if str(int(best)) in values else values.get(str(best), None)
            except Exception:
                return None
        return None
    # find index with closest time
    best_idx = min(range(len(times)), key=lambda i: abs(times[i] - target))
    if abs(times[best_idx] - target) <= tol:
        return values[best_idx]
    return None

def extract_from_json(path):
    """
    Try several heuristics to extract per-player values at SEC seconds.
    Returns dict: {
      'match_file': filename,
      'p1_minerals@60': val or None, 'p2_minerals@60': ...,
      'p1_vespene@60': ...,
      'p1_supply@60': ...,
      'p1_unit_count@60': ...
    }
    """
    out = {'match_file': os.path.basename(path)}
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            j = json.load(fh)
    except Exception as e:
        print("  failed to load JSON:", path, e)
        return None

    # Heuristic 1: container 'players' with per-player timeseries
    players = None
    if isinstance(j, dict) and 'players' in j:
        players = j['players']
    elif isinstance(j, dict) and 'game' in j and 'players' in j['game']:
        players = j['game']['players']

    # If players is a dict keyed by 'p1','p2' or list, normalize
    if players is not None:
        if isinstance(players, dict):
            # try keys or convert to list
            pkeys = sorted(players.keys())
            try:
                p1 = players[pkeys[0]]
                p2 = players[pkeys[1]] if len(pkeys) > 1 else None
            except Exception:
                p1 = None; p2 = None
        elif isinstance(players, list):
            p1 = players[0] if len(players) > 0 else None
            p2 = players[1] if len(players) > 1 else None
        else:
            p1 = p2 = None

        # attempt to find timeseries fields inside player dicts
        def pull_player(player):
            res = {}
            if player is None:
                return res
            # common patterns
            # pattern A: player['resources']['time'] and ['minerals'], ['vespene']
            try:
                resources = player.get('resources') or {}
                if isinstance(resources, dict) and 'time' in resources and 'minerals' in resources:
                    times = [float(t) for t in resources.get('time', [])]
                    res['minerals'] = pick_value_at_time(times, resources.get('minerals', []))
                    res['vespene'] = pick_value_at_time(times, resources.get('vespene', []))
            except Exception:
                pass
            # pattern B: player['timeseries'] list of dicts with 'time','minerals','vespene','supply'
            try:
                ts = player.get('timeseries') or player.get('timeline') or player.get('history')
                if isinstance(ts, list) and len(ts) > 0 and isinstance(ts[0], dict):
                    times = [float(x.get('time', x.get('t', 0))) for x in ts]
                    minerals = [x.get('minerals', x.get('min', None)) for x in ts]
                    vespene = [x.get('vespene', x.get('gas', None)) for x in ts]
                    supply = [x.get('supply', x.get('s', None)) for x in ts]
                    unit_count = [x.get('unit_count', x.get('units', None)) for x in ts]
                    res['minerals'] = pick_value_at_time(times, minerals)
                    res['vespene'] = pick_value_at_time(times, vespene)
                    res['supply'] = pick_value_at_time(times, supply)
                    res['unit_count'] = pick_value_at_time(times, unit_count)
            except Exception:
                pass
            # pattern C: player may have direct keys like 'minerals_by_time' dict
            try:
                for key in ['minerals_by_time', 'mineralsTime', 'minerals']:
                    val = player.get(key)
                    if isinstance(val, dict):
                        # val keyed by time
                        res['minerals'] = res.get('minerals') or pick_value_at_time([float(k) for k in val.keys()], list(val.values()))
            except Exception:
                pass
            # pattern D: simple direct values at checkpoint (rare)
            try:
                for key in ['minerals_at_60', 'minerals_60', 'minerals60']:
                    if key in player:
                        res['minerals'] = res.get('minerals') or player.get(key)
            except Exception:
                pass
            return res

        p1vals = pull_player(p1)
        p2vals = pull_player(p2)

        out['p1_minerals@60'] = p1vals.get('minerals')
        out['p2_minerals@60'] = p2vals.get('minerals')
        out['p1_vespene@60'] = p1vals.get('vespene')
        out['p2_vespene@60'] = p2vals.get('vespene')
        out['p1_supply@60'] = p1vals.get('supply')
        out['p2_supply@60'] = p2vals.get('supply')
        out['p1_unit_count@60'] = p1vals.get('unit_count')
        out['p2_unit_count@60'] = p2vals.get('unit_count')
        # try race
        try:
            out['p1_race'] = p1.get('race') or p1.get('r') if p1 else None
            out['p2_race'] = p2.get('race') or p2.get('r') if p2 else None
        except Exception:
            out['p1_race'] = out['p2_race'] = None

        return out

    # Heuristic 2: top-level timeseries
    if isinstance(j, dict):
        # look for times, players_xxx arrays
        # This is intentionally conservative: if no pattern matches, return None
        return out  # no structured players found

    return None

def main():
    files = find_files()
    print("Found raw JSON files:", len(files))
    extracted = []
    for p in files:
        info = extract_from_json(p)
        if info:
            extracted.append(info)

    df_ex = pd.DataFrame(extracted)
    print("Extracted raw features for", len(df_ex), "files (rows).")

    # Load base CSV and merge on match_file where possible
    if os.path.exists(BASE_CSV):
        base = pd.read_csv(BASE_CSV)
        print("Loaded base CSV rows:", len(base))
        merged = base.merge(df_ex, on='match_file', how='left')
    else:
        print("Base CSV not found at", BASE_CSV, " — writing raw-only file.")
        merged = df_ex

    # Report counts of filled values
    for col in ['p1_minerals@60','p2_minerals@60','p1_vespene@60','p2_vespene@60','p1_supply@60','p2_supply@60']:
        if col in merged.columns:
            print(col, "non-null:", merged[col].notnull().sum())

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)

if __name__ == "__main__":
    main()