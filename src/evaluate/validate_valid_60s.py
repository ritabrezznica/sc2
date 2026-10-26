import pandas as pd

df = pd.read_csv("playerstats_at_60s_valid.csv")

print("Total rows:", len(df))
print("Unique replays:", df["replay_index"].nunique())
print("Rows per replay:")
print(df.groupby("replay_index").size().value_counts().sort_index())

print("Duplicate replay/player pairs:",
      df.duplicated(["replay_index", "playerId"]).sum())

print("Missing values:", df.isna().sum().sum())
print("Unique loops:", df["loop"].unique().tolist())
print("Players:", sorted(df["playerId"].unique().tolist()))

assert len(df) == 46870
assert df["replay_index"].nunique() == 23435
assert df.groupby("replay_index").size().eq(2).all()
assert df.duplicated(["replay_index", "playerId"]).sum() == 0
assert df["loop"].eq(960).all()

print("\nVALIDATION PASSED")
