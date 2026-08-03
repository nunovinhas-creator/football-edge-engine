"""
Módulo de Cálculo de Edge e Expected Value (EV) do Engine.

Definição oficial de Edge (ver docs/AUDIT_MATEMATICA.md, secção
"Definição Oficial de Edge"):

    edge = prob_model - implied_probability(odd_house)

isto é, a diferença entre a probabilidade estimada pelo modelo e a
probabilidade implícita do mercado (1 / odd). Esta é uma grandeza
diferente do Expected Value (EV):

    ev = (prob_model * odd_house) - 1.0

`calculate_edge` é a ÚNICA implementação oficial de Edge do projeto —
todos os módulos que precisem de Edge devem importar e reutilizar esta
função, em vez de recalcular a fórmula localmente.

Ambas as funções recebem `prob_model` como fração (0.0 - 1.0] e devolvem
o resultado na mesma escala; para apresentação em pontos percentuais,
multiplicar o resultado por 100.
"""


def implied_probability(odd: float) -> float:
    """
    Converte odd decimal em probabilidade implícita (0.0 - 1.0).
    """
    if odd <= 1.0:
        return 0.0

    return round(1.0 / odd, 4)


def calculate_edge(prob_model: float, odd_house: float) -> float:
    """
    Calcula o Edge oficial: a diferença entre a probabilidade do modelo
    e a probabilidade implícita de mercado.

        edge = prob_model - implied_probability(odd_house)

    prob_model:
        Probabilidade do modelo, em fração (0.0 - 1.0].

    odd_house:
        Odd decimal da casa (> 1.0).

    Lança ValueError se `prob_model` ou `odd_house` forem inválidos, em
    vez de devolver um valor sentinela silencioso — foi exatamente essa
    falha silenciosa (chamar esta função com uma probabilidade no lugar
    de uma odd) que causou o bug histórico em `src/engine/market.py`.
    """

    if not (0.0 < prob_model <= 1.0):
        raise ValueError(
            f"prob_model inválido: {prob_model!r}. "
            "Deve ser uma probabilidade em fração, no intervalo (0.0, 1.0]."
        )

    if odd_house <= 1.0:
        raise ValueError(
            f"odd_house inválida: {odd_house!r}. "
            "Deve ser uma odd decimal > 1.0 (não uma probabilidade)."
        )

    return round(prob_model - implied_probability(odd_house), 4)


def edge(prob_model: float, odd_house: float) -> float:
    """
    Wrapper para compatibilidade. Ver calculate_edge().
    """
    return calculate_edge(prob_model, odd_house)


def calculate_ev(prob_model: float, odd_house: float) -> float:
    """
    Calcula o Expected Value (EV) por unidade apostada.

        ev = (prob_model * odd_house) - 1.0

    prob_model:
        probabilidade do modelo em fração (0.0 - 1.0)

    odd_house:
        odd decimal da casa
    """
    if odd_house <= 1.0 or prob_model <= 0:
        return -1.0

    return round((prob_model * odd_house) - 1.0, 4)


def expected_value(prob_model: float, odd_house: float) -> float:
    """
    Alias de EV para compatibilidade com versões anteriores.
    """
    return calculate_ev(prob_model, odd_house)
