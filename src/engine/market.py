from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev
)


def analyze_market(
    odds,
    model_probability
):

    analysis = {}


    for outcome, odd in odds.items():

        market_probability = implied_probability(
            odd
        )


        edge = calculate_edge(
            model_probability,
            market_probability
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
