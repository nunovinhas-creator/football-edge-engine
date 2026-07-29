"""
Módulo de Backtesting do Engine de Apostas.
Simula o desempenho histórico (Yield, ROI, Win Rate, Max Drawdown)
aplicando as regras de decisão, incerteza e gestão de banca do motor.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from src.engine.full_engine import run_pipeline

def generate_synthetic_historical_data(n_samples: int = 300) -> pd.DataFrame:
    """Gera um conjunto de dados histórico simulado se não existir features_v2.csv."""
    np.random.seed(42)
    
    attack_avg = np.random.uniform(20, 60, n_samples)
    shots_avg = np.random.uniform(8, 20, n_samples)
    
    # Criar uma relação lógica realista entre estatísticas e total de remates
    lambda_shots = 6.0 + (attack_avg * 0.12) + (shots_avg * 0.35)
    total_shots = np.random.poisson(lam=lambda_shots)
    
    # Odd justa simulada com margem de casa (overround ~5%)
    prob_real = 1.0 - np.exp(-lambda_shots) * (1.0 + lambda_shots) # aprox Poisson CDF
    prob_real = np.clip(prob_real, 0.2, 0.8)
    odd_house = np.round((1.0 / prob_real) * 0.95, 2)
    odd_house = np.clip(odd_house, 1.40, 3.50)

    df = pd.DataFrame({
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
    return df

def run_backtest(initial_bankroll: float = 1000.0, line: float = 12.5):
    # 1. Carregar Dados
    try:
        df = pd.read_csv('research/pressure_shots/features_v2.csv')
        if 'odd_house' not in df.columns:
            # Simular odds se não existirem no CSV
            np.random.seed(42)
            df['odd_house'] = np.round(np.random.uniform(1.75, 2.30, len(df)), 2)
    except FileNotFoundError:
        print("💡 Ficheiro features_v2.csv não encontrado. A gerar 300 jogos históricos simulados...")
        df = generate_synthetic_historical_data(n_samples=300)

    # Definir Target (Over line)
    df['target_over'] = (df['total_shots'] > line).astype(int)

    feature_cols = [
        'is_home', 'attack_avg_last5', 'dangerous_attack_avg_last5', 
        'ball_safe_avg_last5', 'total_shots_avg_last5', 'shots_on_target_avg_last5',
        'attack_difference', 'dangerous_attack_difference', 'ball_safe_difference'
    ]

    X = df[feature_cols].values
    y = df['target_over'].values
    odds = df['odd_house'].values

    # 2. Out-of-Fold Predictions (5-Fold Cross Validation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df))
    oof_stds = np.zeros(len(df))

    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_tr, y_tr)

        # Probabilidade média e Desvio-Padrão entre árvores do ensemble
        oof_probs[val_idx] = model.predict_proba(X_va)[:, 1]
        tree_probas = np.array([tree.predict_proba(X_va)[:, 1] for tree in model.estimators_])
        oof_stds[val_idx] = np.std(tree_probas, axis=0)

    # 3. Execução Cronológica da Banca
    current_bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0

    bets_analyzed = len(df)
    bets_placed = 0
    bets_won = 0
    total_staked = 0.0
    total_profit = 0.0
    gross_win = 0.0
    gross_loss = 0.0

    print("\n" + "=" * 60)
    print(f" SIMULAÇÃO DE BACKTESTING DO ENGINE (Linha: Over {line})")
    print(f" Banca Inicial: {initial_bankroll:.2f}€ | Amostra: {bets_analyzed} jogos")
    print("=" * 60)

    for i in range(len(df)):
        prob = oof_probs[i]
        std = oof_stds[i]
        odd = odds[i]
        actual_result = y[i]  # 1 se ganhou Over, 0 se perdeu
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
            stake = pipeline_res["stake_amount"]
            
            # Se a banca restante for inferior à stake, ajusta
            stake = min(stake, current_bankroll)
            if stake <= 0:
                continue

            total_staked += stake

            if actual_result == 1:
                # Aposta Ganha
                bets_won += 1
                profit = stake * (odd - 1.0)
                gross_win += profit
                current_bankroll += profit
                total_profit += profit
            else:
                # Aposta Perdida
                gross_loss += stake
                current_bankroll -= stake
                total_profit -= stake

            # Atualizar Peak e Max Drawdown
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            
            dd = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        bankroll_history.append(current_bankroll)

    # 4. Cálculo de Métricas Finais
    win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0.0
    yield_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    roi_pct = ((current_bankroll - initial_bankroll) / initial_bankroll * 100)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    # 5. Relatório Visual de Desempenho
    print(f"\n📊 ESTATÍSTICAS DE APOSATAS:")
    print(f"  • Jogos Analisados   : {bets_analyzed}")
    print(f"  • Apostas Realizadas : {bets_placed} ({bets_placed / bets_analyzed:.1%} de taxa de seleção)")
    print(f"  • Apostas Ganhas     : {bets_won}")
    print(f"  • Taxa de Acerto     : {win_rate:.2f}%")
    
    print(f"\n💰 DESEMPENHO FINANCEIRO:")
    print(f"  • Volume Apostado    : {total_staked:.2f}€")
    print(f"  • Lucro / Prejuízo   : {total_profit:+.2f}€")
    print(f"  • Yield (ROI/Aposta) : {yield_pct:+.2f}%")
    print(f"  • ROI sobre Banca    : {roi_pct:+.2f}%")
    print(f"  • Profit Factor      : {profit_factor:.2f}")
    print(f"  • Max Drawdown (Queda): {max_drawdown * 100:.2f}%")
    print(f"  • Banca Final        : {current_bankroll:.2f}€")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_backtest()
