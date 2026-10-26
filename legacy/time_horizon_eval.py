import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

CHECKPOINTS = [60, 120, 180, 240, 300, 360]
REPLAY_DIR = "data/unpacked"   # matches your step2 extraction location
OUT_CSV = "data/processed/time_horizon_metrics.csv"

def find_p1_result(data):
    # Try common top-level fields
    if isinstance(data, dict):
        if "winner" in data:
            try:
                w = int(data["winner"])
                return 1 if w == 1 else 0
            except Exception:
                pass
        if "p1_result" in data:
            v = str(data["p1_result"]).lower()
            if "win" in v:
                return 1
            if "loss" in v or "lose" in v:
                return 0
        if "p1_won" in data:
            try:
                return int(data["p1_won"])
            except Exception:
                pass

    # Fallback: recursive search for player/result pairs
    stack = [data]
    seen = set()
    while stack:
        node = stack.pop()
        oid = id(node)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(node, dict):
            pid = node.get("playerID", node.get("playerId"))
            result = node.get("result", node.get("Result"))
            if pid is not None and result is not None:
                try:
                    pid_int = int(pid)
                except Exception:
                    pid_int = None
                if pid_int == 1:
                    r = str(result).lower()
                    if "win" in r or "victory" in r:
                        return 1
                    if "loss" in r or "lose" in r:
                        return 0
            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    return None


def extract_features_for_checkpoint(match_data, sec):
    tracker_events = match_data.get("trackerEvents", [])
    game_events = match_data.get("gameEvents", [])

    cutoff_loop = sec * 22.4
    window_tracker = [e for e in tracker_events if e.get("loop", 0) <= cutoff_loop]
    window_game = [e for e in game_events if e.get("loop", 0) <= cutoff_loop]

    p1_workers = p2_workers = 0
    p1_buildings = p2_buildings = 0
    worker_names = {"drone", "scv", "probe"}
    for e in window_tracker:
        evt_type = str(e.get("evtTypeName", ""))
        unit_name = str(e.get("unitTypeName", "") or e.get("unitType", "")).lower()
        p_id = e.get("playerId", e.get("controlPlayerId", e.get("upkeepPlayerId", -1)))

        if evt_type in ["UnitBorn", "UnitInit", "UnitDone"]:
            if any(w in unit_name for w in worker_names):
                if p_id in [1, 0]:
                    p1_workers += 1
                elif p_id in [2, 1]:
                    p2_workers += 1

            if "building" in unit_name or unit_name.endswith("townhall") or unit_name.endswith("gateway") or unit_name.endswith("baracks"):
                if p_id in [1, 0]:
                    p1_buildings += 1
                elif p_id in [2, 1]:
                    p2_buildings += 1

    p1_actions = 0
    p2_actions = 0
    for e in window_game:
        user_id = None
        if isinstance(e.get("userid", {}), dict):
            user_id = e.get("userid", {}).get("userId")
        if user_id is None:
            user_id = e.get("userId", None)
        if user_id == 0:
            p1_actions += 1
        elif user_id == 1:
            p2_actions += 1

    p1_apm = round(p1_actions / max(sec / 60.0, 1e-9), 2)
    p2_apm = round(p2_actions / max(sec / 60.0, 1e-9), 2)

    features = {
        "p1_apm": p1_apm,
        "p1_workers": p1_workers,
        "p1_buildings": p1_buildings,
        "delta_apm": round(p1_apm - p2_apm, 2),
        "delta_workers": p1_workers - p2_workers,
        "delta_buildings": p1_buildings - p2_buildings,
    }
    return features


def load_all_matches():
    # look for .SC2Replay.json files
    files = sorted(glob.glob(os.path.join(REPLAY_DIR, "*.SC2Replay.json")))
    if not files:
        raise FileNotFoundError(f"No .SC2Replay.json files found in {REPLAY_DIR}")

    rows = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except Exception as ex:
            print("Failed to load", f, ex)
            continue

        y = find_p1_result(data)
        if y is None:
            # skip unlabeled
            continue

        ckpt_features = {}
        for sec in CHECKPOINTS:
            feat = extract_features_for_checkpoint(data, sec)
            for k, v in feat.items():
                ckpt_features[f"{k}@{sec}"] = v

        row = {"match_file": os.path.basename(f), "win": int(y)}
        row.update(ckpt_features)
        rows.append(row)

    if not rows:
        raise ValueError(f"No usable labeled matches found in {REPLAY_DIR}")

    return pd.DataFrame(rows)


def evaluate_checkpoint(df, checkpoint):
    feature_cols = [
        f"p1_apm@{checkpoint}",
        f"p1_workers@{checkpoint}",
        f"p1_buildings@{checkpoint}",
        f"delta_apm@{checkpoint}",
        f"delta_workers@{checkpoint}",
        f"delta_buildings@{checkpoint}",
    ]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["win"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, briers = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            gamma=0,
            min_child_weight=1,
            random_state=42,
            use_label_encoder=False,
            verbosity=0
        )

        model.fit(X.iloc[train_idx], y.iloc[test_idx])
        preds = model.predict(X.iloc[test_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]

        accs.append(accuracy_score(y.iloc[test_idx], preds))
        aucs.append(roc_auc_score(y.iloc[test_idx], probs))
        briers.append(brier_score_loss(y.iloc[test_idx], probs))

    return {
        "checkpoint_seconds": checkpoint,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers)),
    }


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    df = load_all_matches()
    print(f"Loaded {len(df)} usable matches")
    print(df[["match_file", "win"]].head())

    metrics = []
    for sec in CHECKPOINTS:
        try:
            metrics.append(evaluate_checkpoint(df, sec))
        except Exception as e:
            print(f"Failed checkpoint {sec}: {e}")
            metrics.append({
                "checkpoint_seconds": sec,
                "acc_mean": np.nan,
                "acc_std": np.nan,
                "auc_mean": np.nan,
                "auc_std": np.nan,
                "brier_mean": np.nan,
                "brier_std": np.nan,
            })

    out = pd.DataFrame(metrics)
    out.to_csv(OUT_CSV, index=False)

    print("\n=== Time-Horizon CV Summary ===")
    print(out.to_string(index=False))
    print(f"\nSaved metrics to {OUT_CSV}")