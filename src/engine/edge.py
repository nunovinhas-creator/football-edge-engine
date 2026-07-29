"""
Módulo de Cálculo de Edge (Expected Value - EV) do Engine.
"""

def calculate_edge(prob_model: float, odd_house: float) -> float:
    """
    Calcula o Edge (Valor Esperado) de uma aposta.
    
    :param prob_model: Probabilidade estimada pelo modelo (0.0 a 1.0).
    :param odd_house: Odd oferecida pela casa de apostas (> 1.0).
    :return: Edge percentual em formato decimal (ex: 0.05 para +5% EV).
    """
    if odd_house <= 1.0 or prob_model <= 0.0:
        return -1.0
        
    ev = (prob_model * odd_house) - 1.0
    return round(ev, 4)

def edge(prob_model: float, odd_house: float) -> float:
    """Wrapper para compatibilidade."""
    return calculate_edge(prob_model, odd_house)
