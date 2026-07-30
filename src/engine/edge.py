"""
Módulo de Cálculo de Edge (Expected Value - EV) do Engine.
"""


def implied_probability(odd: float) -> float:
    """
    Converte odd decimal em probabilidade implícita.
    """
    if odd <= 1.0:
        return 0.0

    return round(1.0 / odd, 4)


def calculate_edge(prob_model: float, odd_house: float) -> float:
    """
    Calcula o Edge (Expected Value).

    prob_model:
        Probabilidade do modelo (0.0 - 1.0)

    odd_house:
        Odd decimal da casa
    """

    if odd_house <= 1.0 or prob_model <= 0.0:
        return -1.0

    ev = (prob_model * odd_house) - 1.0

    return round(ev, 4)


def edge(prob_model: float, odd_house: float) -> float:
    """
    Wrapper para compatibilidade.
    """
    return calculate_edge(prob_model, odd_house)


def expected_value(prob_model: float, odd_house: float) -> float:
    """
    Alias de EV para compatibilidade com versões anteriores.
    """
    return calculate_edge(prob_model, odd_house)


def calculate_ev(prob_model: float, odd_house: float) -> float:
    """
    Compatibilidade com versões anteriores.
    Calcula Expected Value.
    
    prob_model:
        probabilidade do modelo em decimal (0-1)

    odd_house:
        odd da casa
    """
    if odd_house <= 1.0 or prob_model <= 0:
        return -1.0

    return round((prob_model * odd_house) - 1.0, 4)
