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
    """Cria a tabela de histórico se não existir."""
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
            goal_in_next_15m BOOLEAN DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_snapshot(match_data: Dict[str, Any]):
    """Insere um novo snapshot de pressão de um jogo a decorrer."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO match_snapshots (
            timestamp, match_id, home_team, away_team, current_minute,
            home_score, away_score, dangerous_attacks_10m, shots_on_target_10m,
            corners_10m, live_odd_over
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        str(match_data['match_id']),
        match_data['home_team'],
        match_data['away_team'],
        match_data['current_minute'],
        match_data['home_score'],
        match_data['away_score'],
        match_data['dangerous_attacks_10m'],
        match_data['shots_on_target_10m'],
        match_data['corners_10m'],
        match_data.get('live_odd_over', 1.85)
    ))
    
    conn.commit()
    conn.close()

def update_outcomes(match_id: str, current_minute: int, goal_occurred: bool):
    """
    Atualiza snapshots antigos (registados há 15 minutos) com o resultado real.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Atualiza registos entre minute-18 e minute-12 para dar margem à janela de 15m
    cursor.execute("""
        UPDATE match_snapshots
        SET goal_in_next_15m = ?
        WHERE match_id = ? 
          AND current_minute BETWEEN ? AND ?
          AND goal_in_next_15m IS NULL
    """, (goal_occurred, match_id, current_minute - 18, current_minute - 12))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"✅ Base de dados do Logger pronta em: {DB_PATH}")
