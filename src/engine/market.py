from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev
)


def analyze_market(
    odds,
    model_probability
):
    """
    Analisa um conjunto de odds de mercado (dict outcome -> odd) face à
    probabilidade do modelo.

    model_probability:
        Probabilidade do modelo, em fração (0.0 - 1.0].

    Nota (correção de bug): anteriormente esta função chamava
    `calculate_edge(model_probability, market_probability)`, passando a
    probabilidade implícita do mercado (0.0-1.0) onde `calculate_edge`
    espera uma odd decimal (> 1.0). Como quase toda a probabilidade é
    <= 1.0, o edge devolvido era sistematicamente -1.0. A chamada correta
    passa a própria odd (`odd`), tal como já acontecia para `calculate_ev`.
    """

    analysis = {}


    for outcome, odd in odds.items():

        market_probability = implied_probability(
            odd
        )


        edge = calculate_edge(
            model_probability,
            odd
        )


        ev = calculate_ev(
            model_probability,
            odd
        )


        analysis[outcome] = {
            "odd": odd,
            "market_probability": market_probability,
            "edge": edge,
            "ev": ev
        }


    return analysis
