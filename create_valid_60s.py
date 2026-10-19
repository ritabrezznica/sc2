import pandas as pd

source = "playerstats_at_60s.csv"
output = "playerstats_at_60s_valid.csv"

df = pd.read_csv(source)

# Keep only genuine 60-second snapshots
df["loop"] = pd.to_numeric(df["loop"], errors="coerce")
df = df[df["loop"] == 960].copy()

# Keep only complete 1v1 replays: exactly players 1 and 2
valid_replays = []

for replay, group in df.groupby("replay_index"):
    players = set(group["playerId"].astype(str))
    if players == {"1", "2"} and len(group) == 2:
        valid_replays.append(replay)

clean = df[df["replay_index"].isin(valid_replays)].copy()
clean.to_csv(output, index=False)

print("Original rows:", len(pd.read_csv(source)))
print("Valid 60-second rows:", len(clean))
print("Valid 1v1 replays:", clean["replay_index"].nunique())
print("Rows removed:", len(pd.read_csv(source)) - len(clean))
print("Saved as:", output)
print("Loops remaining:", sorted(clean["loop"].unique()))
