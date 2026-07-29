"""
Script Principal de Monitorização Live de Golos Iminentes.
"""

import time
from src.api.live_fetcher import BSDLiveFetcher
from src.live.engine import LiveGoalEngine
from src.utils.telegram_notifier import send_telegram_alert

def run_live_pipeline():
    print("\n🚀 A iniciar varredura em tempo real dos jogos...")
    
    fetcher = BSDLiveFetcher()
    engine = LiveGoalEngine()
    
    events = fetcher.get_live_events()
    
    if not events:
        print("ℹ️ Nenhum jogo a decorrer neste momento na BSD API.")
        return

    alerts_sent = 0

    for event in events:
        match_data = fetcher.parse_live_metrics_for_engine(event)
        
        # Calcula a probabilidade de golo nos próximos 15 minutos
        p_goal_15m = engine.predict_next_goal_probability(match_data)
        
        home = match_data['home_team']
        away = match_data['away_team']
        min_curr = match_data['current_minute']
        score = f"{match_data['home_score']}-{match_data['away_score']}"
        
        print(f"🏟️ {home} {score} {away} ({min_curr}') -> P(Golo 15m): {p_goal_15m*100:.1f}%")
        
        # Threshold de Alerta Live: Dispara se a probabilidade de golo for >= 65%
        if p_goal_15m >= 0.65:
            msg = (
                f"🔥 *ALERTA LIVE: PRESSÃO EXTREMA / GOLO IMINENTE*\n"
                f"⚽ *{home} vs {away}*\n"
                f"⏱️ *Resultado / Minuto:* {score} (`{min_curr}'`)\n"
                f"📊 *Prob. Golo (Próx. 15m):* `{p_goal_15m*100:.1f}%`\n"
                f"📈 *Ataques Perigosos (10m):* `{match_data['dangerous_attacks_10m']}`\n"
                f"🎯 *Remates no Alvo (10m):* `{match_data['shots_on_target_10m']}`\n"
                f"💡 *Recomendação:* Mercado Over Golo Live / Próximo Golo\n"
            )
            send_telegram_alert(msg)
            alerts_sent += 1

    print(f"✅ Varredura concluída. Alertas enviados: {alerts_sent}\n")

if __name__ == "__main__":
    run_live_pipeline()
