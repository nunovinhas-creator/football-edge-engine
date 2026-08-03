import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtest.goal_label import recompute_goal_in_next_15m


DB="data/live_history.db"


conn=sqlite3.connect(DB)

cur=conn.cursor()


recompute_goal_in_next_15m(conn)


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
