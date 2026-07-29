"""
Módulo de Backtesting do Engine de Apostas com Stress Testing, Gráfico Visual e Notificação Telegram.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from src.engine.full_engine import run_pipeline
from src.utils.telegram_notifier import send_telegram_alert

def generate_synthetic_historical_data(n_samples: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    attack_avg = np.random.uniform(20, 60, n_samples)
    shots_avg = np.random.uniform(8, 20, n_samples)
    
    lambda_shots = 6.0 + (attack_avg * 0.12) + (shots_avg * 0.35)
    total_shots = np.random.poisson(lam=lambda_shots)
    
    prob_real = 1.0 - np.exp(-lambda_shots) * (1.0 + lambda_shots)
    prob_real = np.clip(prob_real, 0.2, 0.8)
    odd_house = np.round((1.0 / prob_real) * 0.95, 2)
    odd_house = np.clip(odd_house, 1.40, 3.50)

    return pd.DataFrame({
        'match_id': [f"Match_{i+1:03d}" for i in range(n_samples)],
        'is_home': np.random.choice([0, 1], n_samples),
        'attack_avg_last5': attack_avg,
        'dangerous_attack_avg_last5': np.random.uniform(10, 40, n_samples),
        'ball_safe_avg_last5': np.random.uniform(30, 70, n_samples),
        'total_shots_avg_last5': shots_avg,
        'shots_on_target_avg_last5': np.random.uniform(3, 8, n_samples),
        'attack_difference': np.random.uniform(-15, 15, n_samples),
        'dangerous_attack_difference': np.random.uniform(-10, 10, n_samples),
        'ball_safe_difference': np.random.uniform(-20, 20, n_samples),
        'total_shots': total_shots,
        'odd_house': odd_house
    })

def run_backtest(
    initial_bankroll: float = 1000.0, 
    line: float = 12.5,
    slippage_pct: float = 0.02,
    commission_pct: float = 0.00
):
    try:
        df = pd.read_csv('research/pressure_shots/features_v2.csv')
        if 'odd_house' not in df.columns:
            np.random.seed(42)
            df['odd_house'] = np.round(np.random.uniform(1.75, 2.30, len(df)), 2)
    except FileNotFoundError:
        df = generate_synthetic_historical_data(n_samples=300)

    df['target_over'] = (df['total_shots'] > line).astype(int)

    feature_cols = [
        'is_home', 'attack_avg_last5', 'dangerous_attack_avg_last5', 
        'ball_safe_avg_last5', 'total_shots_avg_last5', 'shots_on_target_avg_last5',
        'attack_difference', 'dangerous_attack_difference', 'ball_safe_difference'
    ]

    X = df[feature_cols].values
    y = df['target_over'].values
    odds = df['odd_house'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df))
    oof_stds = np.zeros(len(df))

    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_tr, y_tr)

        oof_probs[val_idx] = model.predict_proba(X_va)[:, 1]
        tree_probas = np.array([tree.predict_proba(X_va)[:, 1] for tree in model.estimators_])
        oof_stds[val_idx] = np.std(tree_probas, axis=0)

    current_bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]
    
    current_bankroll_stressed = initial_bankroll
    bankroll_history_stressed = [initial_bankroll]

    peak_bankroll = initial_bankroll
    max_drawdown = 0.0

    bets_placed = 0
    bets_won = 0
    total_staked = 0.0
    total_profit = 0.0
    total_profit_stressed = 0.0

    for i in range(len(df)):
        prob = oof_probs[i]
        std = oof_stds[i]
        odd = odds[i]
        actual_result = y[i]
        match_id = df['match_id'].iloc[i] if 'match_id' in df.columns else f"Jogo_{i+1}"

        pipeline_res = run_pipeline(
            prob_model=prob,
            odd_house=odd,
            bankroll=current_bankroll,
            sample_size=5,
            model_std=std,
            match_info=f"{match_id}"
        )

        decision = pipeline_res["decision"]

        if decision.action == "BET":
            bets_placed += 1
            stake = min(pipeline_res["stake_amount"], current_bankroll)
            if stake <= 0:
                continue

            total_staked += stake
            odd_stressed = max(1.01, odd * (1.0 - slippage_pct))

            if actual_result == 1:
                bets_won += 1
                profit = stake * (odd - 1.0)
                profit_stressed = (stake * (odd_stressed - 1.0)) * (1.0 - commission_pct)
                
                current_bankroll += profit
                total_profit += profit
                
                current_bankroll_stressed += profit_stressed
                total_profit_stressed += profit_stressed
            else:
                current_bankroll -= stake
                total_profit -= stake
                
                current_bankroll_stressed -= stake
                total_profit_stressed -= stake

            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            
            dd = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        bankroll_history.append(current_bankroll)
        bankroll_history_stressed.append(current_bankroll_stressed)

    win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0.0
    yield_base = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    yield_stressed = (total_profit_stressed / total_staked * 100) if total_staked > 0 else 0.0

    # Imprimir no Terminal
    print("\n" + "=" * 60)
    print(f" RESULTADOS DE BACKTESTING & STRESS TEST")
    print("=" * 60)
    print(f"  • Apostas Aprovadas   : {bets_placed} de {len(df)} ({win_rate:.1f}% acerto)")
    print(f"  • Banca Base Final    : {current_bankroll:.2f}€ (Yield Base: {yield_base:+.2f}%)")
    print(f"  • Banca com Stress    : {current_bankroll_stressed:.2f}€ (Yield com Slippage {slippage_pct:.1%}: {yield_stressed:+.2f}%)")
    print(f"  • Max Drawdown        : {max_drawdown * 100:.2f}%")

    # Gerar Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(bankroll_history, label='Evolução da Banca (Ideal)', color='#2ecc71', linewidth=2)
    plt.plot(bankroll_history_stressed, label=f'Evolução da Banca com Stress ({slippage_pct:.0%} Slippage)', color='#e74c3c', linestyle='--', linewidth=2)
    plt.axhline(y=initial_bankroll, color='#95a5a6', linestyle=':', label='Banca Inicial')
    plt.title('Football Edge Engine - Curva de Evolução da Banca', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Apostas')
    plt.ylabel('Banca (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart_path = 'research/bankroll_curve.png'
    plt.savefig(chart_path, dpi=300)
    print(f"📈 Gráfico salvo com sucesso em: {chart_path}\n" + "=" * 60)

    # Enviar Notificação Resumo para o Telegram
    telegram_msg = (
        f"📊 *RELATÓRIO DE EXECUÇÃO DIÁRIA*\n\n"
        f"⚽ *Jogos Analisados:* {len(df)}\n"
        f"✅ *Apostas Aprovadas:* {bets_placed} ({win_rate:.1f}% acerto)\n"
        f"💰 *Banca Base Final:* {current_bankroll:.2f}€ (`{yield_base:+.2f}% Yield`)\n"
        f"🛡️ *Banca com Stress (2%):* {current_bankroll_stressed:.2f}€ (`{yield_stressed:+.2f}% Yield`)\n"
        f"📉 *Max Drawdown:* {max_drawdown * 100:.2f}%\n\n"
        f"📈 _Gráfico de performance guardado nos artefactos do GitHub._"
    )
    send_telegram_alert(telegram_msg)

if __name__ == "__main__":
    run_backtest()
