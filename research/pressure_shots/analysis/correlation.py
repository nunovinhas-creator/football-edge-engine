import pandas as pd

df = pd.read_csv(
    "data/processed/pressure_shots/history_features.csv"
)

cols = [
    "attack_avg_last5",
    "dangerous_attack_avg_last5",
    "ball_safe_avg_last5",
]

for c in cols:
    corr = df[c].corr(df["total_shots"])
    print(f"{c}: {corr:.3f}")
