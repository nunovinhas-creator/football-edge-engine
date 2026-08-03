"""
Data Logger para registo de métricas Live e validação futura de Backtest.
"""

import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/live_history.db")

def init_db():
    """Cria a tabela de histórico se não existir e adiciona colunas novas em esquemas antigos."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            match_id TEXT,
            home_team TEXT,
            away_team TEXT,
            current_minute INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            dangerous_attacks_10m INTEGER,
            shots_on_target_10m INTEGER,
            corners_10m INTEGER,
            live_odd_over REAL,
            pressure REAL,
            dominance REAL DEFAULT 0,
            estimated_xg REAL DEFAULT 0,
            dominance_index REAL DEFAULT 0,
            estimated_xg_10m REAL DEFAULT 0,
            home_possession REAL DEFAULT 50,
            red_cards INTEGER DEFAULT 0,
            last_goal_minute INTEGER DEFAULT NULL,
            live_xg REAL DEFAULT 0,
            possession REAL DEFAULT 50,
            goal_in_next_15m BOOLEAN DEFAULT NULL
        )
    """)

    existing_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(match_snapshots)")
    }

    for column_name, column_def in [
        ("dominance", "REAL DEFAULT 0"),
        ("estimated_xg", "REAL DEFAULT 0"),
        ("dominance_index", "REAL DEFAULT 0"),
        ("estimated_xg_10m", "REAL DEFAULT 0"),
        ("home_possession", "REAL DEFAULT 50"),
        ("last_goal_minute", "INTEGER DEFAULT NULL"),
        ("red_cards", "INTEGER DEFAULT 0"),
        ("possession", "REAL DEFAULT 50"),
    ]:
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE match_snapshots ADD COLUMN {column_name} {column_def}"
            )

    conn.commit()
    conn.close()

def log_snapshot(match_data: Dict[str, Any]):
    """Insere um novo snapshot de pressão de um jogo a decorrer."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    existing_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(match_snapshots)")
    }

    columns = [
        "timestamp", "match_id", "home_team", "away_team", "current_minute",
        "home_score", "away_score", "dangerous_attacks_10m", "shots_on_target_10m",
        "corners_10m", "live_odd_over", "pressure"
    ]
    values = [
        datetime.utcnow().isoformat(),
        str(match_data["match_id"]),
        match_data["home_team"],
        match_data["away_team"],
        match_data["current_minute"],
        match_data["home_score"],
        match_data["away_score"],
        match_data.get("dangerous_attacks_10m",0),
        match_data.get("shots_on_target_10m",0),
        match_data.get("corners_10m",0),
        match_data.get("live_odd_over",1.85),
        match_data.get("pressure",0),
    ]

    dominance_value = match_data.get("dominance_index", 0)
    estimated_xg_value = match_data.get("estimated_xg_10m", 0)

    for column_name, value in [
        ("dominance", dominance_value),
        ("dominance_index", dominance_value),
        ("estimated_xg", estimated_xg_value),
        ("estimated_xg_10m", estimated_xg_value),
        ("home_possession", match_data.get("home_possession", 50)),
        ("red_cards", match_data.get("red_cards", 0)),
        ("last_goal_minute", match_data.get("last_goal_minute")),
        ("possession", match_data.get("home_possession", 50)),
    ]:
        if column_name in existing_columns:
            columns.append(column_name)
            values.append(value)

    placeholders = ", ".join(["?"] * len(columns))
    cursor.execute(
        f"INSERT INTO match_snapshots ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"✅ Base de dados do Logger pronta em: {DB_PATH}")
