import os
import glob
import json
import math
import pandas as pd

REPLAY_DIR = "replays_json"
OUT_CSV = "data/processed/all_match_features_raw.csv"
BASE_CSV = "data/processed/all_match_features.csv"
SEC = 60.0
TOL = 2.0  # seconds tolerance

def list_replay_files():
    pattern = os.path.join(REPLAY_DIR, "*.json")
    files = glob.glob(pattern)
    # include files that end with .SC2Replay.json as well
    files += glob.glob(os.path.join(REPLAY_DIR, "*.SC2Replay.json"))
    return sorted(files)

def float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def time_from_event(ev):
    # Try several common time keys and conversions
    for k in ("second", "time", "s", "t", "tick", "ms", "timestamp"):
        if k in ev:
            v = ev[k]
            if v is None:
                continue
            # convert ms -> seconds heuristically
            fv = float_or_none(v)
            if fv is None:
                continue
            if k == "ms" and fv > 1000:
                return fv / 1000.0
            if k == "timestamp" and fv > 1000000000:
                # probably epoch ms
                # convert ms->s
                if fv > 1e12:
                    return fv / 1000.0
                return fv
            return fv
    # check nested 'data' or similar
    if "data" in ev and isinstance(ev["data"], dict):
        return time_from_event(ev["data"])
    return None

def maybe_get_fields(d):
    # from a dict d, try to extract minerals/vespene/supply/unit_count/player/playerId
    out = {}
    if not isinstance(d, dict):
        return out
    # common keys
    for k in ("minerals","minerals_current","mineral","m"):
        if k in d and out.get("minerals") is None:
            out["minerals"] = float_or_none(d[k])
    for k in ("vespene","gas","vespene_current","g"):
        if k in d and out.get("vespene") is None:
            out["vespene"] = float_or_none(d[k])
    for k in ("supply","supply_used","s","supply_current"):
        if k in d and out.get("supply") is None:
            out["supply"] = float_or_none(d[k])
    for k in ("unit_count","units","unitTotal","units_count","unit_count_current"):
        if k in d and out.get("unit_count") is None:
            out["unit_count"] = float_or_none(d[k])
    # player identification
    for k in ("player","playerId","player_index","p","pid","playerIndex"):
        if k in d and out.get("player") is None:
            out["player"] = d[k]
    return out

def normalize_player_id(pid):
    # normalize to 1 or 2 if possible
    if pid is None:
        return None
    try:
        i = int(pid)
        # Some files index players as 0/1, others 1/2
        if i in (0,1):
            return 1 if i == 0 else 2
        if i in (1,2):
            return i
        return i
    except Exception:
        # strings like 'Player1' -> extract trailing number
        s = str(pid)
        import re
        m = re.search(r"(\d+)$", s)
        if m:
            return int(m.group(1))
        return None

def extract_from_file(path):
    basename = os.path.basename(path)
    result = {"match_file": basename}
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print("Failed to open JSON:", path, e)
        return None

    # candidates for per-player stats
    pstats = {1: {}, 2: {}}

    # Heuristic A: look for trackerEvents with per-player mappings or player field
    events = []
    for key in ("trackerEvents", "gameEvents", "messageEvents", "additional_information"):
        if key in j and isinstance(j[key], list):
            events.extend(j[key])

    # Also check j.get('trackerEvts') fallback
    if "trackerEvts" in j and isinstance(j["trackerEvts"], list):
        events.extend(j["trackerEvts"])

    for ev in events:
        if not isinstance(ev, dict):
            continue
        t = time_from_event(ev)
        # if event time near checkpoint, try to extract fields
        if t is None or abs(t - SEC) > TOL:
            # there could be nested per-player snapshots keyed by time; check inside ev
            # If not close, skip
            continue

        # attempt to extract per-player maps directly
        # pattern: ev contains 'players' mapping id->dict
        if "players" in ev and isinstance(ev["players"], dict):
            for pid_str, pdata in ev["players"].items():
                pid = normalize_player_id(pid_str)
                if pid is None:
                    continue
                got = maybe_get_fields(pdata)
                for k,v in got.items():
                    if k == "player": continue
                    if v is not None:
                        pstats.setdefault(pid, {})[k] = v
            continue

        # pattern: event contains a 'data' dict with player info
        candidates = []
        if "data" in ev and isinstance(ev["data"], dict):
            candidates.append(ev["data"])
        candidates.append(ev)

        for cand in candidates:
            got = maybe_get_fields(cand)
            if got.get("player") is not None:
                pid = normalize_player_id(got["player"])
                if pid in (1,2):
                    for k,v in got.items():
                        if k=="player": continue
                        if v is not None:
                            pstats.setdefault(pid, {})[k] = v
                    continue
            # else, if cand has per-player numeric keys (like '1', '2')
            for subk, subv in list(cand.items()):
                if str(subk).isdigit():
                    pid = normalize_player_id(subk)
                    if isinstance(subv, dict):
                        got2 = maybe_get_fields(subv)
                        for k,v in got2.items():
                            if v is not None:
                                pstats.setdefault(pid, {})[k] = v

    # Heuristic B: check details -> players for simple values
    try:
        det = j.get("details") or {}
        players = det.get("players") or det.get("playerList") or det.get("PlayerList")
        if isinstance(players, list) and len(players) >= 2:
            # try to map first two entries to p1/p2
            for idx in (0,1):
                pdata = players[idx] or {}
                pid = idx + 1
                got = maybe_get_fields(pdata)
                for k,v in got.items():
                    if v is not None:
                        pstats.setdefault(pid, {})[k] = v
                # try race if present
                for rk in ("race", "r"):
                    if rk in pdata:
                        pstats.setdefault(pid, {})["race"] = pdata[rk]
        elif isinstance(players, dict):
            # if dict keyed by player id
            for pid_str, pdata in players.items():
                pid = normalize_player_id(pid_str)
                if pid is None:
                    continue
                got = maybe_get_fields(pdata)
                for k,v in got.items():
                    if v is not None:
                        pstats.setdefault(pid, {})[k] = v
    except Exception:
        pass

    # Heuristic C: check initData or metadata for static info
    try:
        init = j.get("initData") or {}
        # some files put human-readable player order here
        if isinstance(init, dict):
            # try to find players mapping with race or name
            ips = init.get("players") or init.get("playerList")
            if isinstance(ips, list) and len(ips) >= 2:
                for idx in (0,1):
                    pdata = ips[idx] or {}
                    for rk in ("race","r"):
                        if rk in pdata:
                            pstats.setdefault(idx+1, {})["race"] = pdata[rk]
    except Exception:
        pass

    # Map collected stats into output fields
    for pid in (1,2):
        stats = pstats.get(pid, {})
        result[f"p{pid}_minerals@{int(SEC)}"] = stats.get("minerals")
        result[f"p{pid}_vespene@{int(SEC)}"] = stats.get("vespene")
        result[f"p{pid}_supply@{int(SEC)}"] = stats.get("supply")
        result[f"p{pid}_unit_count@{int(SEC)}"] = stats.get("unit_count")
        result[f"p{pid}_race"] = stats.get("race")

    return result

def main():
    files = list_replay_files()
    print("Found replay JSON files:", len(files))
    extracted = []
    for p in files:
        info = extract_from_file(p)
        if info:
            extracted.append(info)
    df_ex = pd.DataFrame(extracted)
    print("Extracted rows:", len(df_ex))
    # Load base CSV if present and merge
    if os.path.exists(BASE_CSV):
        base = pd.read_csv(BASE_CSV)
        print("Loaded base CSV rows:", len(base))
        merged = base.merge(df_ex, on="match_file", how="left")
    else:
        merged = df_ex
    # Print counts
    for col in [f"p{pid}_minerals@{int(SEC)}" for pid in (1,2)] + [f"p{pid}_vespene@{int(SEC)}" for pid in (1,2)] + [f"p{pid}_supply@{int(SEC)}" for pid in (1,2)]:
        if col in merged.columns:
            print(col, "non-null:", merged[col].notnull().sum())
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)

if __name__ == "__main__":
    main()