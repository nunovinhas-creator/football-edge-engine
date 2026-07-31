import sqlite3


DB="data/live_history.db"


conn=sqlite3.connect(DB)

cur=conn.cursor()


cur.execute("""
UPDATE match_snapshots
SET goal_in_next_15m = 1
WHERE id IN (

SELECT a.id

FROM match_snapshots a

JOIN match_snapshots b

ON a.match_id=b.match_id

AND b.timestamp > a.timestamp

AND b.timestamp <= datetime(a.timestamp,'+15 minutes')

WHERE 
b.home_score != a.home_score
OR
b.away_score != a.away_score

)
""")


conn.commit()

print(
"labels positivos:",
cur.rowcount
)


conn.close()
