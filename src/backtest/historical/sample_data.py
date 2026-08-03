"""
Gerador de um conjunto de jogos históricos sintéticos, usado nos testes de
integração e no exemplo de execução do Backtesting Framework
(`src/tools/run_backtest_example.py`).

Reutiliza deliberadamente peças já existentes do motor — `DecisionEngine`
(src/engine/decision.py) para produzir a "decisão do motor" e
`apply_market_conditions` (src/backtest/market.py) para simular a odd de
mercado a partir da probabilidade real — em vez de inventar lógica nova,
para que o dataset sintético se pareça o mais possível com dados reais
produzidos pelo sistema. Os dados são inteiramente sintéticos (não há
jogos reais no repositório) — servem apenas para demonstrar o framework.
"""

import random
from datetime import date, timedelta
from typing import Any, Dict, List

from src.backtest.market import apply_market_conditions
from src.engine.decision import DecisionEngine

COMPETITIONS = ["Primeira Liga", "La Liga", "Premier League"]
MARKETS = ["HOME", "DRAW", "AWAY", "OVER_2.5", "UNDER_2.5"]
TEAMS = [
    "Benfica", "Porto", "Sporting", "Braga", "Real Madrid", "Barcelona",
    "Atletico Madrid", "Sevilla", "Man City", "Liverpool", "Arsenal", "Chelsea",
]


def generate_sample_dataset(n_games: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Gera `n_games` apostas históricas sintéticas mas plausíveis:
        - probabilidade "real" (desconhecida do modelo) por jogo;
        - probabilidade do modelo = probabilidade real + ruído (leve
          sobreconfiança), simulando um modelo imperfeito mas informativo;
        - odd de mercado derivada da probabilidade real via
          `apply_market_conditions` (margem + slippage), tal como um
          bookmaker real cotaria;
        - decisão do motor calculada pelo `DecisionEngine` real;
        - resultado real simulado com a probabilidade "real" do jogo.
    """
    rng = random.Random(seed)
    engine = DecisionEngine(max_kelly_fraction=0.25, min_edge=3.0)

    start = date(2025, 8, 1)
    rows = []

    for i in range(n_games):
        home, away = rng.sample(TEAMS, 2)
        market = rng.choice(MARKETS)
        competition = rng.choice(COMPETITIONS)
        home_or_away = "HOME" if market in ("HOME", "OVER_2.5") else rng.choice(["HOME", "AWAY"])

        true_prob = rng.uniform(0.25, 0.75)
        # Ruído de calibração: o modelo tende a ser levemente sobreconfiante.
        noise = rng.gauss(0.03, 0.05)
        model_prob = min(max(true_prob + noise, 0.02), 0.98)

        fair_odd = 1.0 / true_prob
        market_odd = apply_market_conditions(fair_odd, margin=0.05, slippage=0.0)

        recommendation = engine.evaluate_bet(market, model_prob * 100.0, market_odd)

        won = rng.random() < true_prob

        rows.append(
            {
                "jogo": f"{home} vs {away}",
                "data": (start + timedelta(days=i)).isoformat(),
                "mercado": market,
                "odd": market_odd,
                "probabilidade": round(model_prob, 4),
                "decisao": recommendation.action,
                "resultado": "WIN" if won else "LOSS",
                "competicao": competition,
                "venue": home_or_away,
            }
        )

    return rows
