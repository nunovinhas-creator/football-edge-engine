from typing import Optional

import numpy as np
from scipy.stats import poisson

from src.engine.kelly import calculate_adaptive_kelly_fraction
from src.engine.kelly import kelly_fraction as _kelly_fraction

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
    max_stake_pct: float = 0.02,
    lambda_tier: Optional[str] = None,
    effective_sample_size: Optional[float] = None,
) -> float:
    """
    Calcula a percentagem da banca a apostar usando Fractional Kelly (default 1/4 Kelly)
    com um limite máximo (Cap/Hard Limit) por aposta.

    Reutiliza `src.engine.kelly.kelly_fraction` para o Kelly completo e
    `src.engine.kelly.calculate_adaptive_kelly_fraction` (Melhoria #6) para
    escalar `fraction` pela confiança do modelo — não recalcula a fórmula
    de Kelly localmente (elimina a duplicação face a `src/engine/kelly.py`).

    `lambda_tier`/`effective_sample_size` são opcionais: omissos, o
    resultado é exatamente igual ao de antes desta melhoria.
    """
    if odds <= 1.0 or prob_win <= 0:
        return 0.0

    kelly_full = _kelly_fraction(prob_win, odds)

    if kelly_full <= 0:
        return 0.0

    # Aplicar Fractional Kelly (escalada pela confiança, se disponível) e o Hard Limit (Cap)
    adaptive_fraction = calculate_adaptive_kelly_fraction(
        fraction, lambda_tier, effective_sample_size
    )
    kelly_fractional = kelly_full * adaptive_fraction
    final_stake = min(kelly_fractional, max_stake_pct)

    return round(final_stake, 4)
