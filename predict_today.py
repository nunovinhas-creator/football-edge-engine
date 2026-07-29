"""
Script de Previsão Diária com BSD API (BzzoiroClient) e Relatório Consolidado.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.api.client import BzzoiroClient
from src.engine.full_engine import run_pipeline
from src.utils.telegram_notifier import send_telegram_alert

def fetch_live_data_from_bsd():
    print("📡 A ligar à BSD API via BzzoiroClient...")
    client = BzzoiroClient()
    
    try:
        response = client.get("odds/?limit=100&offset=0")
        results = response.get("results", [])
        
        if not results:
            print("ℹ️ Nenhum evento retornado pela BSD API no momento.")
            return pd.DataFrame()

        matches = []
        for item in results:
            matches.append({
                'match_id': item.get('event_id', 'Unknown'),
                'home_team': item.get('home_team', 'Equipa A'),
                'away_team': item.get('away_team', 'Equipa B'),
                'is_home': 1,
                'attack_avg_last5': item.get('attack_avg', 45.0),
                'dangerous_attack_avg_last5': item.get('dangerous_attack_avg', 30.0),
                'ball_safe_avg_last5': item.get('ball_safe_avg', 50.0),
                'total_shots_avg_last5': item.get('total_shots_avg', 14.0),
                'shots_on_target_avg_last5': item.get('shots_on_target_avg', 5.0),
                'attack_difference': item.get('attack_diff', 0.0),
                'dangerous_attack_difference': item.get('dangerous_attack_diff', 0.0),
                'ball_safe_difference': item.get('ball_safe_diff', 0.0),
                'odd_house': float(item.get('price', item.get('odd', 2.00)))
            })

        return pd.DataFrame(matches)

    except Exception as e:
        print(f"❌ Erro ao comunicar com a BSD API: {e}")
        return pd.DataFrame()

def main():
    print("⚽ A iniciar motor de previsão diária...")

    # 1. Carregar Histórico
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

    # 2. Obter Jogos da BSD API
    df_today = fetch_live_data_from_bsd()

    if df_today.empty:
        send_telegram_alert("ℹ️ *Execução Diária:* Nenhum jogo ativo encontrado na BSD API.")
        return

    # 3. Fazer Previsões
    X_today = df_today[feature_cols].values
    probs = model.predict_proba(X_today)[:, 1]
    
    tree_probas = np.array([tree.predict_proba(X_today)[:, 1] for tree in model.estimators_])
    stds = np.std(tree_probas, axis=0)

    # 4. Avaliar Apostas
    current_bankroll = 1000.0
    approved_bets = []

    for i in range(len(df_today)):
        match_row = df_today.iloc[i]
        prob = probs[i]
        std = stds[i]
        odd = match_row['odd_house']
        match_name = f"{match_row['home_team']} vs {match_row['away_team']}"

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
                'match': match_name,
                'id': match_row['match_id'],
                'odd': odd,
                'prob': prob,
                'stake': res["stake_amount"]
            })

    # 5. Enviar Relatório Consolidado no Telegram
    if approved_bets:
        # Ordenar pelas apostas de maior valor/stake
        approved_bets.sort(key=lambda x: x['stake'], reverse=True)
        top_bets = approved_bets[:5]

        msg = (
            f"📊 *ANÁLISE DIÁRIA BSD API*\n\n"
            f"⚽ *Jogos Analisados:* {len(df_today)}\n"
            f"🎯 *Total de Apostas EV+:* {len(approved_bets)}\n\n"
            f"🔥 *TOP 5 MELHORES APOSTAS:*\n"
            f"────────────────────\n"
        )

        for b in top_bets:
            msg += (
                f"⚽ *{b['match']}*\n"
                f"📈 Mercado: Over {line} Remates\n"
                f"💰 Odd BSD: `{b['odd']:.2f}` | Prob: `{b['prob']*100:.1f}%`\n"
                f"💵 Stake: *{b['stake']:.2f}€*\n"
                f"────────────────────\n"
            )

        print(f"✅ Previsões concluídas! A enviar resumo das {len(top_bets)} melhores apostas para o Telegram...")
        send_telegram_alert(msg)
    else:
        send_telegram_alert(f"ℹ️ *Análise Diária:* {len(df_today)} jogos analisados, mas nenhuma oportunidade cumpriu os critérios do Kelly Criterion.")

if __name__ == "__main__":
    main()
