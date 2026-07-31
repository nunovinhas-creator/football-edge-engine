import sqlite3
import pandas as pd


DB="data/live_history.db"

conn=sqlite3.connect(DB)


query="""

SELECT
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


df.to_csv(
"data/training_dataset.csv",
index=False
)


print("Dataset criado")
