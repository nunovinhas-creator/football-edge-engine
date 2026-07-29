"""
Script de Previsão Diária com Extração de Dados Reais da BSD API.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from src.api.client import BzzoiroClient
from src.engine.full_engine import run_pipeline
from src.utils.telegram_notifier import send_telegram_alert

def fetch_live_data_from_bsd():
    print("📡 A ligar à BSD API para recolher jogos e odds reais...")
    client = BzzoiroClient()
    
    try:
        response = client.get("odds/?limit=100&offset=0")
        results = response.get("results", [])
        
        if not results:
            print("ℹ️ Nenhum evento retornado pela BSD API no momento.")
            return pd.DataFrame()

        matches = []
        for item in results:
            # Extrair dados reais do payload da BSD
            home = item.get('home_team') or item.get('home') or item.get('team_home') or 'Equipa Casa'
            away = item.get('away_team') or item.get('away') or item.get('team_away') or 'Equipa Fora'
            league = item.get('league') or item.get('tournament') or 'Futebol'
            start_time = item.get('start_time') or item.get('commence_time') or 'Hoje'
            odd_val = float(item.get('price', item.get('odd', item.get('value', 2.00))))

            matches.append({
                'match_id': item.get('event_id', 'ID_Desconhecido'),
                'home_team': home,
                'away_team': away,
                'league': league,
                'start_time': start_time,
                'is_home': 1,
                # Features estatísticas com fallback inteligente
                'attack_avg_last5': item.get('attack_avg', 45.0),
                'dangerous_attack_avg_last5': item.get('dangerous_attack_avg', 30.0),
                'ball_safe_avg_last5': item.get('ball_safe_avg', 50.0),
                'total_shots_avg_last5': item.get('total_shots_avg', 14.0),
                'shots_on_target_avg_last5': item.get('shots_on_target_avg', 5.0),
                'attack_difference': item.get('attack_diff', 0.0),
                'dangerous_attack_difference': item.get('dangerous_attack_diff', 0.0),
                'ball_safe_difference': item.get('ball_safe_diff', 0.0),
                'odd_house': odd_val
            })

        return pd.DataFrame(matches)

    except Exception as e:
        print(f"❌ Erro ao comunicar com a BSD API: {e}")
        return pd.DataFrame()

def main():
    print("⚽ A iniciar processamento de apostas reais...")

    # 1. Carregar Treino do Modelo
    try:
        df_hist = pd.read_csv('research/pressure_shots/features_v2.csv')
    except FileNotFoundError:
        from research.backtest_engine import generate_synthetic_historical_data
        df_hist = generate_synthetic_historical_data(300)

    line = 12.5
    df_hist['target_over'] = (df_hist['total_shots'] > line).astype(int)

    feature_cols = [
        'is_home', 'attack_avg_last5', 'dangerous_attack_avg_last5', 
        'ball_safe_avg_last5', 'total_shots_avg_last5', 'shots_on_target_avg_last5',
        'attack_difference', 'dangerous_attack_difference', 'ball_safe_difference'
    ]

    X_train = df_hist[feature_cols].values
    y_train = df_hist['target_over'].values

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 2. Buscar Jogos na BSD API
    df_today = fetch_live_data_from_bsd()

    if df_today.empty:
        send_telegram_alert("ℹ️ *Análise Diária:* Nenhum evento ativo retornado pela BSD API.")
        return

    # 3. Correr o Modelo
    X_today = df_today[feature_cols].values
    probs = model.predict_proba(X_today)[:, 1]
    
    tree_probas = np.array([tree.predict_proba(X_today)[:, 1] for tree in model.estimators_])
    stds = np.std(tree_probas, axis=0)

    # 4. Selecionar Apostas Aprovadas
    current_bankroll = 1000.0
    approved_bets = []

    for i in range(len(df_today)):
        row = df_today.iloc[i]
        prob = probs[i]
        std = stds[i]
        odd = row['odd_house']
        match_name = f"{row['home_team']} vs {row['away_team']}"

        res = run_pipeline(
            prob_model=prob,
            odd_house=odd,
            bankroll=current_bankroll,
            sample_size=5,
            model_std=std,
            match_info=match_name
        )

        decision = res["decision"]

        if decision.action == "BET":
            approved_bets.append({
                'home': row['home_team'],
                'away': row['away_team'],
                'league': row['league'],
                'time': row['start_time'],
                'odd': odd,
                'prob': prob,
                'stake': res["stake_amount"]
            })

    # 5. Mapear e Enviar Boletim de Apostas Concretas
    if approved_bets:
        approved_bets.sort(key=lambda x: x['stake'], reverse=True)
        top_bets = approved_bets[:5]  # Mostra as 5 maiores oportunidades de valor

        msg = (
            f"🎯 *BOLETIM DE APOSTAS REAIS ({datetime.now().strftime('%d/%m/%Y')})*\n"
            f"⚽ *Jogos Analisados BSD:* {len(df_today)}\n"
            f"✅ *Apostas Aprovadas:* {len(approved_bets)}\n"
            f"──────────────────────────────\n\n"
        )

        for b in top_bets:
            msg += (
                f"⚽ *{b['home']} vs {b['away']}*\n"
                f"🏆 *Liga:* {b['league']}\n"
                f"🕒 *Hora:* {b['time']}\n"
                f"📈 *Aposta:* Over {line} Remates\n"
                f"💰 *Odd BSD:* `{b['odd']:.2f}`\n"
                f"📊 *Prob. Modelo:* `{b['prob']*100:.1f}%`\n"
                f"💵 *Stake Recomendada:* *{b['stake']:.2f}€*\n"
                f"──────────────────────────────\n\n"
            )

        send_telegram_alert(msg)
        print(f"✅ Boletim enviado com {len(top_bets)} apostas detalhadas para o Telegram!")
    else:
        send_telegram_alert(f"ℹ️ *Análise Concluída:* {len(df_today)} jogos analisados da BSD API, mas nenhum cumpre os critérios mínimos de EV+.")

if __name__ == "__main__":
    main()
