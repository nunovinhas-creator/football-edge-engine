import pandas as pd


df = pd.read_csv(
    "data/processed/pressure_shots/history_features.csv"
)


df["event_date"] = pd.to_datetime(df["event_date"])


df = df.sort_values(
    ["team_id", "event_date"]
)


# médias históricas da própria equipa
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
        .transform(
            lambda x: x.shift(1)
            .rolling(5, min_periods=1)
            .mean()
        )
    )


# diferenças contra adversário

df["attack_difference"] = (
    df["attack_avg_last5"] -
    df["opp_attack"]
)


df["dangerous_attack_difference"] = (
    df["dangerous_attack_avg_last5"] -
    df["opp_dangerous_attack"]
)


df["ball_safe_difference"] = (
    df["ball_safe_avg_last5"] -
    df["opp_ball_safe"]
)


# guardar

output = (
    "data/processed/pressure_shots/features_v2.csv"
)


df.to_csv(
    output,
    index=False
)


print("Features v2 criadas")
print(df.shape)
print(df.columns.tolist())
