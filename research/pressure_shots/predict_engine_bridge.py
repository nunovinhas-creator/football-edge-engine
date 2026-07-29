"""
Ponte entre o Modelo de ML (Research) e o Engine de Tomada de Decisão (src/engine).
Calcula a probabilidade, a incerteza do modelo (std entre árvores) e executa o pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.engine.full_engine import run_pipeline

def run_bridge_pipeline(line: float = 12.5, bankroll: float = 1000.0):
    # 1. Tentar carregar features
    try:
        df = pd.read_csv('research/pressure_shots/features_v2.csv')
    except FileNotFoundError:
        print("Ficheiro features_v2.csv não encontrado. Gerando dados de demonstração...")
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            'is_home': np.random.choice([0, 1], n_samples),
            'attack_avg_last5': np.random.uniform(20, 60, n_samples),
            'dangerous_attack_avg_last5': np.random.uniform(10, 40, n_samples),
            'ball_safe_avg_last5': np.random.uniform(30, 70, n_samples),
            'total_shots_avg_last5': np.random.uniform(8, 20, n_samples),
            'shots_on_target_avg_last5': np.random.uniform(3, 8, n_samples),
            'attack_difference': np.random.uniform(-15, 15, n_samples),
            'dangerous_attack_difference': np.random.uniform(-10, 10, n_samples),
            'ball_safe_difference': np.random.uniform(-20, 20, n_samples),
            'total_shots': np.random.poisson(lam=13, size=n_samples)
        })

    # 2. Definir Target Binário (Over/Under)
    df['target_over'] = (df['total_shots'] > line).astype(int)

    feature_cols = [
        'is_home', 'attack_avg_last5', 'dangerous_attack_avg_last5', 
        'ball_safe_avg_last5', 'total_shots_avg_last5', 'shots_on_target_avg_last5',
        'attack_difference', 'dangerous_attack_difference', 'ball_safe_difference'
    ]

    X = df[feature_cols]
    y = df['target_over']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Treinar RandomForest
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Obter Probabilidades Globais
    probas = clf.predict_proba(X_test)[:, 1]

    # 5. Obter Incerteza (Desvio Padrão da previsão entre as 100 árvores)
    tree_probas = np.array([tree.predict_proba(X_test.values)[:, 1] for tree in clf.estimators_])
    stds = np.std(tree_probas, axis=0)

    print(f"\n==================================================")
    print(f" AVALIAÇÃO DE JOGOS NO ENGINE (Linha: Over {line})")
    print(f"==================================================\n")

    # 6. Simular odds da casa e correr no Engine para os primeiros 5 jogos de teste
    mock_odds = [1.95, 2.10, 1.80, 2.25, 1.90]
    bet_count = 0

    for i in range(min(5, len(X_test))):
        prob = probas[i]
        std = stds[i]
        odd = mock_odds[i % len(mock_odds)]
        match_name = f"Jogo #{X_test.index[i]} (Over {line} Remates)"

        res = run_pipeline(
            prob_model=prob,
            odd_house=odd,
            bankroll=bankroll,
            sample_size=5,
            model_std=std,
            match_info=match_name
        )

        print(res["explanation"])
        if res["decision"].action == "BET":
            bet_count += 1
            print(f"💰 Recomendação: Apostar {res['stake_amount']}€ @ Odd {odd}\n")
        else:
            print(f"🚫 Aposta Rejeitada pelo Engine.\n")

    print(f"Resumo da Sessão: {bet_count} entradas aprovadas de {min(5, len(X_test))} analisadas.")

if __name__ == "__main__":
    run_bridge_pipeline()
