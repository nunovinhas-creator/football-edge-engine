import sqlite3


DB="data/live_history.db"


conn=sqlite3.connect(DB)

cur=conn.cursor()


cur.execute("""
UPDATE match_snapshots
SET goal_in_next_15m =
(
SELECT 
CASE
WHEN EXISTS
(
SELECT 1
FROM match_snapshots b
WHERE 
b.match_id = match_snapshots.match_id
AND b.home_score + b.away_score >
match_snapshots.home_score + match_snapshots.away_score
AND b.current_minute > match_snapshots.current_minute
AND b.current_minute <= match_snapshots.current_minute + 15
)
THEN 1
ELSE 0
END
)
WHERE current_minute IS NOT NULL
""")


conn.commit()

print("labels recalculadas")

print(
cur.execute("""
SELECT goal_in_next_15m,count(*)
FROM match_snapshots
GROUP BY goal_in_next_15m
""").fetchall()
)


conn.close()
