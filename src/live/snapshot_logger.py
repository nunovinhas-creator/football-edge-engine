
import sqlite3
from datetime import datetime


DB="data/live_history.db"


def save_snapshot(data):

    conn=sqlite3.connect(DB)

    cur=conn.cursor()


    cur.execute("""
    INSERT INTO match_snapshots
    (
    timestamp,
    match_id,
    home_team,
    away_team,
    minute,
    home_score,
    away_score,
    pressure,
    dangerous_attacks,
    shots_on_target,
    corners,
    xg_live,
    odds_over25,
    red_cards
    )

    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)

    """,

    (

    datetime.utcnow().isoformat(),

    data["match_id"],

    data["home_team"],

    data["away_team"],

    data["minute"],

    data["home_score"],

    data["away_score"],

    data["pressure"],

    data.get("dangerous_attacks",0),

    data.get("shots_on_target",0),

    data.get("corners",0),

    data.get("xg_live",0),

    data.get("odds_over25",0),

    data.get("red_cards",0)

    ))


    conn.commit()

    conn.close()


