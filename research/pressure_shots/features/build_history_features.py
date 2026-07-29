import pandas as pd

df = pd.read_csv("data/processed/pressure_shots/raw_team_match.csv")

df["event_date"] = pd.to_datetime(df["event_date"])
df = df.sort_values(["team_id", "event_date"])

cols = [
    "attack",
    "dangerous_attack",
    "ball_safe",
    "total_shots",
    "shots_on_target",
]

for c in cols:
    df[f"{c}_avg_last5"] = (
        df.groupby("team_id")[c]
          .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

print(
    df[
        [
            "team_id",
            "event_date",
            "attack",
            "attack_avg_last5",
            "dangerous_attack",
            "dangerous_attack_avg_last5",
            "total_shots",
            "total_shots_avg_last5",
        ]
    ].head(15)
)
import os

os.makedirs("data/processed/pressure_shots", exist_ok=True)
df.to_csv("data/processed/pressure_shots/history_features.csv", index=False)

print("Ficheiro gravado em data/processed/pressure_shots/history_features.csv")
