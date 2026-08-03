from src.engine.edge import implied_probability, edge, expected_value


def analyze_bet(odd, model_probability):
    """
    model_probability:
        Probabilidade do modelo em escala percentual (0-100), a mesma
        convenção usada por Match.probability / predict_probability().
    """

    market_probability = implied_probability(odd)

    prob_fraction = model_probability / 100.0

    if odd <= 1.0 or prob_fraction <= 0.0 or prob_fraction > 1.0:
        return {
            "odd": odd,
            "model_probability": model_probability,
            "market_probability": round(market_probability * 100, 2),
            "edge": -100.0,
            "ev": -100.0,
            "decision": "PASS"
        }

    # Edge (%) — implementação oficial e única (src/engine/edge.py)
    bet_edge = round(edge(prob_fraction, odd) * 100, 2)
    ev_fraction = expected_value(prob_fraction, odd)

    if bet_edge >= 5 and ev_fraction >= 0.08:
        decision = "VALUE BET"
    elif bet_edge >= 3:
        decision = "WATCH"
    else:
        decision = "PASS"

    return {
        "odd": odd,
        "model_probability": model_probability,
        "market_probability": round(market_probability * 100, 2),
        "edge": bet_edge,
        "ev": round(ev_fraction * 100, 2),
        "decision": decision
    }
