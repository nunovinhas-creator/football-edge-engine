"""
Fetch Upcoming Matches — próximas 48 horas.

Script standalone que vai buscar à BSD API os jogos agendados para as
próximas 48 horas e imprime-os na consola, ordenados cronologicamente.
Reutiliza `src.report.upcoming_matches.fetch_upcoming_events` (a mesma
função já usada pelo painel "Oportunidades das Próximas 24 Horas") apenas
com uma janela temporal diferente (48h em vez de 24h) — nenhuma lógica de
negócio nova, nenhum endpoint novo.
"""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.report.dashboard_data import extract_competition
from src.report.upcoming_matches import fetch_upcoming_events

WINDOW_HOURS = 48


def main():
    dated_events = fetch_upcoming_events(hours=WINDOW_HOURS)

    print(f"\n📅 Jogos agendados nas próximas {WINDOW_HOURS}h: {len(dated_events)}\n")

    if not dated_events:
        print("❄️ Nenhum jogo agendado neste período.")
        return

    for event, kickoff in dated_events:
        competition = extract_competition(event)
        home = event.get("home_team", "Casa")
        away = event.get("away_team", "Fora")
        kickoff_label = kickoff.strftime("%Y-%m-%d %H:%M UTC") if kickoff else "—"

        print(f"{kickoff_label}  |  {competition}  |  {home} vs {away}  (id={event.get('id')})")


if __name__ == "__main__":
    main()
