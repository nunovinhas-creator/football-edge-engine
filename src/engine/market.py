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

    Remoção do overround (Melhoria #7 da auditoria matemática): `odds` já
    é, por definição desta função, o conjunto completo das opções de um
    mesmo mercado (ex. as 3 odds de um 1X2, ou as 2 odds de um Over/Under
    ou BTTS) — exatamente o que `calculate_edge()` precisa como
    `market_odds` para remover o overround e usar a probabilidade fair em
    vez da implícita. `market_probability` mantém-se a probabilidade
    implícita simples (com margem), inalterada, para não quebrar
    consumidores existentes deste campo.
    """

    analysis = {}


    for outcome, odd in odds.items():

        market_probability = implied_probability(
            odd
        )


        edge = calculate_edge(
            model_probability,
            odd,
            market_odds=odds
        )


        ev = calculate_ev(
            model_probability,
            odd,
            market_odds=odds
        )


        analysis[outcome] = {
            "odd": odd,
            "market_probability": market_probability,
            "edge": edge,
            "ev": ev
        }


    return analysis
