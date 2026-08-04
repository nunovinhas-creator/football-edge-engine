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
from src.live.pressure import PressureEngine
from src.live.engine import LiveGoalEngine
from src.live.providers.api_odds_provider import APIOddsProvider
from src.engine.live_decision import evaluate_live_market
from src.live.value_alerts import notify_if_value
from src.models.live_state import LiveMatchState
from src.backtest.logger import init_db, log_snapshot

def run_live_pipeline():
    init_db()
    print("\n🚀 [AUTOMATIC RUN] A iniciar varredura de jogos em direto...")

    try:
        fetcher = BSDLiveFetcher()
    except Exception as e:
        print(f"⚠️ Erro ao inicializar o fetcher (verificar credenciais/mock): {e}")
        return

    try:
        odds_provider = APIOddsProvider()
    except Exception as e:
        print(f"⚠️ Odds provider indisponível (alertas +EV desativados nesta run): {e}")
        odds_provider = None

    engine = LiveGoalEngine()
    events = fetcher.get_live_events()
    
    if not events:
        print("ℹ️ Nenhum jogo a decorrer neste momento.")
        return

    print(f"⚽ {len(events)} jogos em direto identificados.")

    for event in events:
        match_data = fetcher.parse_live_metrics_for_engine(event)

        if match_data.get("home_score") is None or match_data.get("away_score") is None:
            print(
                f"⚠️ Jogo ignorado (match_id={match_data.get('match_id')}): "
                "home_score/away_score ausente (None)."
            )
            continue

        # A label goal_in_next_15m é recalculada para toda a tabela pelo
        # passo "4.5 Recalcular Labels" do workflow (create_labels.py,
        # que reutiliza src.backtest.goal_label) — não há recalculo
        # incremental aqui.

        # Previsão do Motor
        pressure = PressureEngine.score(
            minute=match_data.get("current_minute",0),
            home_score=match_data.get("home_score",0),
            away_score=match_data.get("away_score",0),
            last_goal_minute=match_data.get("last_goal_minute"),
            odds_over=match_data.get("odds_over")
        )
        live_state = LiveMatchState(
            minute=match_data.get("current_minute", 0),
            home_score=match_data.get("home_score", 0),
            away_score=match_data.get("away_score", 0),

            home_xg_last5=match_data.get("home_xg_last5", 1.5),
            away_conceded_xg_last5=match_data.get("away_conceded_xg_last5", 1.2),

            home_style=match_data.get("home_style", "balanced"),

            dangerous_attacks_10m=match_data.get("dangerous_attacks_10m", 0),
            shots_on_target_10m=match_data.get("shots_on_target_10m", 0),
            shots_10m=match_data.get("shots_10m", 0),
            corners_10m=match_data.get("corners_10m", 0),

            possession=match_data.get("home_possession", 50.0),

            previous_pressure=pressure,

            goals_last_15=match_data.get("goals_last_15", 0),
            last_goal_minute=match_data.get("last_goal_minute"),
            red_cards=match_data.get("red_cards", 0),

            game_state="live"
        )

        prediction = engine.predict_next_goal_probability(live_state)

        if isinstance(prediction, dict):
            snapshot = dict(match_data)
            snapshot["pressure"] = prediction["pressure"]
            snapshot["dominance"] = prediction["dominance_index"]
            snapshot["dominance_index"] = prediction["dominance_index"]
            snapshot["estimated_xg"] = prediction["estimated_xg_10m"]
            snapshot["estimated_xg_10m"] = prediction["estimated_xg_10m"]
            log_snapshot(snapshot)
            p_goal_15m = (
                prediction.get("next_goal_probability", 0.0) / 100
            )
        else:
            p_goal_15m = prediction

        home = match_data['home_team']
        away = match_data['away_team']
        min_curr = match_data['current_minute']
        score = f"{match_data['home_score']}-{match_data['away_score']}"

        print(f"🏟️ {home} {score} {away} ({min_curr}') -> P(Golo 15m): {p_goal_15m*100:.1f}%")

        if odds_provider is not None and isinstance(prediction, dict):
            try:
                odds_response = odds_provider.get_live_odds(match_data["match_id"])
                bookie_odd = odds_response["odds"]["over_15_goals"]

                decision = evaluate_live_market(
                    probability_pct=prediction.get("next_goal_probability", 0.0),
                    bookie_odd=bookie_odd,
                    market="NEXT GOAL (15m)"
                )

                if notify_if_value(
                    match_id=match_data["match_id"],
                    home_team=home,
                    away_team=away,
                    minute=min_curr,
                    score=score,
                    decision=decision
                ):
                    print(f"📲 Alerta +EV enviado para {home} vs {away}.")
            except Exception as e:
                print(f"⚠️ Não foi possível avaliar valor/odds para match_id={match_data.get('match_id')}: {e}")

if __name__ == "__main__":
    run_live_pipeline()
