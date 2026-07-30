"""
Script Principal de Monitorização e Logging Automático em Tempo Real.
"""

import os
import sys
from pathlib import Path

# Garantir sys.path correto
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.api.live_fetcher import BSDLiveFetcher
from src.live.engine import LiveGoalEngine
from src.backtest.logger import init_db, log_snapshot, update_outcomes

def run_live_pipeline():
    init_db()
    print("\n🚀 [AUTOMATIC RUN] A iniciar varredura de jogos em direto...")
    
    try:
        fetcher = BSDLiveFetcher()
    except Exception as e:
        print(f"⚠️ Erro ao inicializar o fetcher (verificar credenciais/mock): {e}")
        return

    engine = LiveGoalEngine()
    events = fetcher.get_live_events()
    
    if not events:
        print("ℹ️ Nenhum jogo a decorrer neste momento.")
        return

    print(f"⚽ {len(events)} jogos em direto identificados.")

    for event in events:
        match_data = fetcher.parse_live_metrics_for_engine(event)
        
        # 1. Registar o Snapshot na Base de Dados do Backtest
        log_snapshot(match_data)
        
        # 2. Atualizar se existiu golo nas entradas registadas há ~15 minutos
        # (Consideramos se o placard mudou em relação ao snapshot anterior)
        goal_just_happened = match_data.get('goal_occurred_recently', False)
        update_outcomes(
            match_id=str(match_data['match_id']),
            current_minute=match_data['current_minute'],
            goal_occurred=goal_just_happened
        )

        # 3. Previsão do Motor
        p_goal_15m = engine.predict_next_goal_probability(match_data)
        home = match_data['home_team']
        away = match_data['away_team']
        min_curr = match_data['current_minute']
        score = f"{match_data['home_score']}-{match_data['away_score']}"
        
        print(f"🏟️ {home} {score} {away} ({min_curr}') -> P(Golo 15m): {p_goal_15m*100:.1f}%")

if __name__ == "__main__":
    run_live_pipeline()
