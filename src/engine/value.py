import numpy as np
from src.engine.dixon_coles import dixon_coles_simulate_match, calculate_fractional_kelly

def evaluate_match_value(lambda_home: float, mu_away: float, odds_home: float, odds_draw: float, odds_away: float, rho: float = -0.05):
    """
    Avalia o valor das apostas 1X2 usando a distribuição Dixon-Coles e gestão de banca Fractional Kelly.
    """
    # 1. Gerar matriz de probabilidades Dixon-Coles
    matrix = dixon_coles_simulate_match(lambda_home, mu_away, rho=rho)
    
    # 2. Somar probabilidades por mercado
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.trace(matrix))
    p_away = float(np.sum(np.triu(matrix, 1)))
    
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
