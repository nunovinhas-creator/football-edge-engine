"""
Alertas Telegram para valor (+EV) detetado em jogos ao vivo.

Reutiliza a lógica oficial de edge (src.engine.live_decision) e o
notifier Telegram já usado pelo boletim diário (src.utils.telegram_notifier).
Garante no máximo um alerta por jogo (deduplicado em live_history.db)
para não inundar o Telegram em execuções repetidas do monitor.
"""

import sqlite3

from src.backtest.logger import DB_PATH, init_db
from src.utils.telegram_notifier import send_telegram_alert


def _init_alerts_table():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telegram_value_alerts (
            match_id TEXT PRIMARY KEY,
            sent_at DATETIME DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def already_alerted(match_id) -> bool:
    _init_alerts_table()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT 1 FROM telegram_value_alerts WHERE match_id = ?",
        (str(match_id),)
    ).fetchone()
    conn.close()
    return row is not None


def mark_alerted(match_id):
    _init_alerts_table()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO telegram_value_alerts (match_id) VALUES (?)",
        (str(match_id),)
    )
    conn.commit()
    conn.close()


def notify_if_value(match_id, home_team, away_team, minute, score, decision) -> bool:
    """Envia um alerta Telegram no máximo uma vez por jogo quando o motor
    de decisão (src.engine.live_decision.evaluate_live_market) reporta
    valor (+EV)."""

    if decision.action != "🔥 BET VALUE":
        return False

    if already_alerted(match_id):
        return False

    message = (
        f"🚀 *+EV DETETADO*\n"
        f"⚽ *{home_team} {score} {away_team}*\n"
        f"⏱️ Minuto: {minute}'\n"
        f"🎯 Mercado: {decision.market}\n"
        f"📊 Probabilidade: {decision.probability}%\n"
        f"💰 Odd: {decision.odd}\n"
        f"📈 Edge: {decision.edge}%"
    )

    if send_telegram_alert(message):
        mark_alerted(match_id)
        return True

    return False
