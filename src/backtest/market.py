def apply_market_conditions(raw_odd: float, margin: float = 0.05, slippage: float = 0.02) -> float:
    """
    Aplica a margem da casa de apostas (Vig) e o slippage (atraso na colocação) à odd real.
    """
    if raw_odd <= 1.0:
        return 0.0
    
    # 1. Converter odd justa em probabilidade justa
    fair_prob = 1.0 / raw_odd
    
    # 2. Adicionar a margem (Vig) da casa de apostas
    implied_prob_with_margin = fair_prob / (1.0 - margin)
    bookmaker_odd = 1.0 / implied_prob_with_margin
    
    # 3. Simular slippage (odd cai antes de conseguirmos apostar)
    executable_odd = bookmaker_odd * (1.0 - slippage)
    
    return round(max(executable_odd, 1.01), 2)
