import numpy as np
from src.engine.dixon_coles import dixon_coles_simulate_match, calculate_fractional_kelly


def _market_probabilities_from_matrix(matrix):
    """
    Soma a matriz de resultados exatos do Dixon-Coles por mercado 1X2.
    Extraído para função própria apenas para ser reutilizado por
    `estimate_pregame_probabilities()` sem duplicar a lógica de soma —
    o cálculo em si (tril/trace/triu) não foi alterado.
    """
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.trace(matrix))
    p_away = float(np.sum(np.triu(matrix, 1)))
    return p_home, p_draw, p_away


def estimate_pregame_probabilities(lambda_home: float, mu_away: float, rho: float = -0.05):
    """
    Devolve as probabilidades 1X2 (fração 0.0-1.0) do modelo Dixon-Coles já
    existente, sem exigir odds — ao contrário de `evaluate_match_value()`,
    que também calcula EV/stake e por isso precisa das 3 odds em
    simultâneo. Usada pela pipeline de pré-jogo (`src/collector/client.py`)
    sempre que só é preciso a probabilidade do modelo, não o EV.

    Não recalcula nem duplica a matriz Dixon-Coles: chama
    `dixon_coles_simulate_match()` (inalterado) tal como
    `evaluate_match_value()`.
    """
    matrix = dixon_coles_simulate_match(lambda_home, mu_away, rho=rho)
    p_home, p_draw, p_away = _market_probabilities_from_matrix(matrix)

    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away
    }


def evaluate_match_value(lambda_home: float, mu_away: float, odds_home: float, odds_draw: float, odds_away: float, rho: float = -0.05):
    """
    Avalia o valor das apostas 1X2 usando a distribuição Dixon-Coles e gestão de banca Fractional Kelly.
    """
    # 1. Gerar matriz de probabilidades Dixon-Coles
    matrix = dixon_coles_simulate_match(lambda_home, mu_away, rho=rho)

    # 2. Somar probabilidades por mercado
    p_home, p_draw, p_away = _market_probabilities_from_matrix(matrix)

    # 3. Calcular EV (Expected Value) e Stakes (1/4 Kelly, max 2%)
    results = {
        "home": {
            "prob": p_home,
            "odd": odds_home,
            "ev": (p_home * odds_home) - 1.0,
            "stake_pct": calculate_fractional_kelly(p_home, odds_home)
        },
        "draw": {
            "prob": p_draw,
            "odd": odds_draw,
            "ev": (p_draw * odds_draw) - 1.0,
            "stake_pct": calculate_fractional_kelly(p_draw, odds_draw)
        },
        "away": {
            "prob": p_away,
            "odd": odds_away,
            "ev": (p_away * odds_away) - 1.0,
            "stake_pct": calculate_fractional_kelly(p_away, odds_away)
        }
    }
    
    return results
