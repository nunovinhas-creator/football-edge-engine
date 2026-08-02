import sqlite3
import pandas as pd


DB="data/live_history.db"

conn=sqlite3.connect(DB)


PREVIOUS_COLUMNS=[
"current_minute",
"home_score",
"away_score",
"dangerous_attacks_10m",
"shots_on_target_10m",
"corners_10m",
"live_odd_over",
"pressure",
"live_xg",
"red_cards",
"possession",
"goal_in_next_15m",
]


query="""

SELECT
match_id,
current_minute,
home_score,
away_score,
dangerous_attacks_10m,
shots_on_target_10m,
corners_10m,
live_odd_over,
pressure,
live_xg,
red_cards,
possession,
goal_in_next_15m

FROM match_snapshots

WHERE
current_minute IS NOT NULL
AND goal_in_next_15m IS NOT NULL

"""


df=pd.read_sql(query,conn)

conn.close()


print(df.head())
print(df["goal_in_next_15m"].value_counts())


added_columns=[col for col in df.columns if col not in PREVIOUS_COLUMNS]

print(f"Numero de colunas antes: {len(PREVIOUS_COLUMNS)}")
print(f"Numero de colunas depois: {len(df.columns)}")
print(f"Coluna(s) adicionada(s): {added_columns}")


df.to_csv(
"data/training_dataset.csv",
index=False
)


print("Dataset criado")
