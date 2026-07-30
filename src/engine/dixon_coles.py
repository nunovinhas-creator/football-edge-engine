import numpy as np
from scipy.stats import poisson

def tau(x: int, y: int, lambda_home: float, mu_away: float, rho: float) -> float:
    """
    Fator de correção de Dixon-Coles para ajustar a dependência em scores baixos (0-0, 1-0, 0-1, 1-1).
    """
    if x == 0 and y == 0:
        return 1.0 - (lambda_home * mu_away * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu_away * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_home * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0

def dixon_coles_simulate_match(lambda_home: float, mu_away: float, rho: float = -0.05, max_goals: int = 8):
    """
    Gera a matriz de probabilidades para o resultado exato do jogo ajustada por Dixon-Coles.
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = poisson.pmf(h, lambda_home)
            p_a = poisson.pmf(a, mu_away)
            t = tau(h, a, lambda_home, mu_away, rho)
            matrix[h, a] = p_h * p_a * t

    # Normalizar para garantir que a soma da matriz é exatamente 1.0
    matrix /= np.sum(matrix)
    return matrix

def calculate_fractional_kelly(
    prob_win: float, 
    odds: float, 
    fraction: float = 0.25, 
    max_stake_pct: float = 0.02
) -> float:
    """
    Calcula a percentagem da banca a apostar usando Fractional Kelly (default 1/4 Kelly)
    com um limite máximo (Cap/Hard Limit) por aposta.
    """
    if odds <= 1.0 or prob_win <= 0:
        return 0.0
    
    b = odds - 1.0
    q = 1.0 - prob_win
    
    # Kelly Padrão: f* = (b*p - q) / b
    kelly_full = (b * prob_win - q) / b
    
    if kelly_full <= 0:
        return 0.0
    
    # Aplicar Fractional Kelly e o Hard Limit (Cap)
    kelly_fractional = kelly_full * fraction
    final_stake = min(kelly_fractional, max_stake_pct)
    
    return round(final_stake, 4)
