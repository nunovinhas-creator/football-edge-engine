"""
Script standalone para recalcular `goal_in_next_15m` manualmente.

Reutiliza a implementação oficial (`src.backtest.goal_label`) em vez de
uma lógica própria baseada em `timestamp` — essa versão divergia da
definição oficial (baseada em `current_minute`) e nunca era invocada por
nenhum workflow ou módulo (ver `docs/AUDIT_MATEMATICA.md`, secção 10.2).
Mantido apenas como ponto de entrada manual equivalente a
`src/training/create_labels.py`.
"""

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


rowcount = recompute_goal_in_next_15m(conn)


conn.commit()

print(
"labels recalculadas:",
rowcount
)


conn.close()
