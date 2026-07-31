import sys
import time
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.models.live_state import LiveMatchState
from src.collector.live_fetcher import LiveDataCollector
from src.report.dashboard import render_live_dashboard

def simulate_live_match_stream():
    collector = LiveDataCollector()
    
    # Simula a evolução de um jogo dos 70' aos 75' minutos
    live_ticks = [
        {
            "minute": 70,
            "payload": {
                "fixture": {"status": {"elapsed": 70}},
                "statistics": {"home": {"possession": "58%", "shots_on_target": 2, "total_shots": 5, "corners": 3, "dangerous_attacks": 8}}
            },
            "odd_over15": 1.95
        },
        {
            "minute": 73,
            "payload": {
                "fixture": {"status": {"elapsed": 73}},
                "statistics": {"home": {"possession": "63%", "shots_on_target": 4, "total_shots": 8, "corners": 5, "dangerous_attacks": 13}}
            },
            "odd_over15": 2.10
        },
        {
            "minute": 76,
            "payload": {
                "fixture": {"status": {"elapsed": 76}},
                "statistics": {"home": {"possession": "67%", "shots_on_target": 6, "total_shots": 11, "corners": 7, "dangerous_attacks": 17}}
            },
            "odd_over15": 2.30
        },
    ]

    print("🚀 A iniciar Football Edge Engine Live Feed...\n")
    time.sleep(1)

    for tick in live_ticks:
        state = collector.parse_api_football_to_state(tick["payload"])
        render_live_dashboard("Sporting CP", "Braga", "0-0", state, bookie_over15_odd=tick["odd_over15"])
        print("\n" + "="*80 + "\n")
        time.sleep(2) # Pausa de 2 segundos entre updates de minutos

if __name__ == "__main__":
    simulate_live_match_stream()
