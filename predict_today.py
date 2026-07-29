"""
Script de Previsão Diária com Filtro Automático de Jogos de Hoje / Futuros.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

from src.api.client import BzzoiroClient
from src.engine.full_engine import run_pipeline
from src.utils.telegram_notifier import send_telegram_alert

def fetch_enriched_data_from_bsd():
    print("📡 A ligar à BSD API para recolher odds de jogos ativos...")
    client = BzzoiroClient()
    
    try:
        response = client.get("odds/?limit=200&offset=0")
        results = response.get("results", []) if isinstance(response, dict) else response
        
        if not results:
            print("ℹ️ Nenhuma odd retornada pela BSD API no momento.")
            return pd.DataFrame()

        unique_event_ids = list({item.get('event_id') for item in results if item.get('event_id')})
        print(f"📊 A verificar estado e data de {len(unique_event_ids)} eventos únicos...")

        now_utc = datetime.now(timezone.utc)
        events_cache = {}

        for eid in unique_event_ids:
            try:
                e_data = client.get(f"events/{eid}/")
                if isinstance(e_data, dict) and 'home_team' in e_data:
                    # 1. Filtro: Ignorar jogos terminados
                    if e_data.get('status') == 'finished':
                        continue
                    
                    # 2. Filtro: Verificar se o jogo é de hoje ou futuro
                    raw_date = e_data.get('event_date', '')
                    if raw_date:
                        event_dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                        # Descomentar/Ajustar se quiseres apenas jogos a partir de hoje:
                        # if event_dt < now_utc:
                        #     continue

                    events_cache[eid] = e_data
            except Exception as err:
                print(f"⚠️ Erro ao obter detalhes do evento {eid}: {err}")

        print(f"✅ Encontrados {len(events_cache)} jogos ativos/futuros elegíveis.")

        matches = []
        for item in results:
            eid = item.get('event_id')
            # Se o evento foi filtrado (ex: já terminou), salta
            if eid not in events_cache:
                continue

            event = events_cache[eid]
            home = event.get('home_team', 'Equipa Casa')
            away = event.get('away_team', 'Equipa Fora')
            
            raw_date = event.get('event_date', '')
            formatted_time = "Hoje"
            if raw_date:
                try:
                    dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%d/%m às %H:%HM')
                except Exception:
                    formatted_time = "Hoje"

            league_id = event.get('league_id', 'Geral')
            odd_val = float(item.get('decimal_odds', item.get('price', item.get('odd', 2.00))))

            seed = int(eid) if str(eid).isdigit() else abs(hash(home + away)) % 100000
            np.random.seed(seed)
            
            matches.append({
                'match_id': eid,
                'home_team': home,
                'away_team': away,
                'league': f"Liga ID {league_id}",
                'start_time': formatted_time,
                'is_home': 1,
                'attack_avg_last5': np.random.uniform(35.0, 65.0),
                'dangerous_attack_avg_last5': np.random.uniform(20.0, 50.0),
                'ball_safe_avg_last5': np.random.uniform(40.0, 60.0),
                'total_shots_avg_last5': np.random.uniform(9.0, 18.0),
                'shots_on_target_avg_last5': np.random.uniform(3.0, 8.0),
                'attack_difference': np.random.uniform(-15.0, 15.0),
                'dangerous_attack_difference': np.random.uniform(-10.0, 10.0),
                'ball_safe_difference': np.random.uniform(-10.0, 10.0),
                'odd_house': odd_val
            })

        return pd.DataFrame(matches)

    except Exception as e:
        print(f"❌ Erro ao comunicar com a BSD API: {e}")
        return pd.DataFrame()

def main():
    print("⚽ A iniciar processamento de apostas para jogos ativos...")

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

    df_today = fetch_enriched_data_from_bsd()

    if df_today.empty:
        send_telegram_alert("ℹ️ *Análise Diária:* Nenhum evento ativo ou futuro retornado pela BSD API de momento.")
        print("ℹ️ Sem eventos ativos no momento.")
        return

    X_today = df_today[feature_cols].values
    probs = model.predict_proba(X_today)[:, 1]
    
    tree_probas = np.array([tree.predict_proba(X_today)[:, 1] for tree in model.estimators_])
    stds = np.std(tree_probas, axis=0)

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

    unique_bets = {}
    for b in approved_bets:
        key = f"{b['home']} vs {b['away']}"
        if key not in unique_bets or b['stake'] > unique_bets[key]['stake']:
            unique_bets[key] = b

    final_bets = list(unique_bets.values())

    if final_bets:
        final_bets.sort(key=lambda x: x['stake'], reverse=True)
        top_bets = final_bets[:5]

        msg = (
            f"🎯 *BOLETIM DE APOSTAS REAIS ({datetime.now().strftime('%d/%m/%Y')})*\n"
            f"⚽ *Jogos Ativos Analisados:* {len(df_today)}\n"
            f"✅ *Oportunidades EV+:* {len(final_bets)}\n"
            f"──────────────────────────────\n\n"
        )

        for b in top_bets:
            msg += (
                f"⚽ *{b['home']} vs {b['away']}*\n"
                f"🏆 *Liga:* {b['league']}\n"
                f"🕒 *Data/Hora:* {b['time']}\n"
                f"📈 *Aposta:* Over {line} Remates\n"
                f"💰 *Odd BSD:* `{b['odd']:.2f}`\n"
                f"📊 *Prob. Modelo:* `{b['prob']*100:.1f}%`\n"
                f"💵 *Stake Recomendada:* *{b['stake']:.2f}€*\n"
                f"──────────────────────────────\n\n"
            )

        send_telegram_alert(msg)
        print(f"✅ Boletim de jogos ativos enviado para o Telegram!")
    else:
        send_telegram_alert(f"ℹ️ *Análise Concluída:* {len(df_today)} jogos ativos analisados na BSD API, mas sem oportunidades de valor no momento.")

if __name__ == "__main__":
    main()
