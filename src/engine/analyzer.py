from src.engine.edge import implied_probability, edge, expected_value


def analyze_bet(odd, model_probability):

    market_probability = implied_probability(odd)
    bet_edge = edge(model_probability, odd)
    ev = expected_value(model_probability, odd)

    if bet_edge >= 5 and ev >= 0.08:
        decision = "VALUE BET"
    elif bet_edge >= 3:
        decision = "WATCH"
    else:
        decision = "PASS"

    return {
        "odd": odd,
        "model_probability": model_probability,
        "market_probability": round(market_probability, 2),
        "edge": round(bet_edge, 2),
        "ev": round(ev * 100, 2),
        "decision": decision
    }
