from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev
)

from src.engine.decision import make_decision
from src.engine.stake import calculate_stake


def analyze_match(match):

    odds = match.odds


    results = []


    for outcome, odd in odds.items():

        market_probability = implied_probability(
            odd
        )


        edge = calculate_edge(
            match.probability,
            market_probability
        )


        ev = calculate_ev(
            match.probability,
            odd
        )


        decision = make_decision(
            edge,
            ev
        )


        confidence = (
            "HIGH"
            if match.confidence >= 80
            else
            "MEDIUM"
            if match.confidence >= 60
            else
            "LOW"
        )


        stake = calculate_stake(
            edge,
            confidence
        )


        results.append({

            "match": f"{match.home} vs {match.away}",
            "league": match.league,
            "market": outcome,
            "odd": odd,
            "model_probability": match.probability,
            "market_probability": market_probability,
            "edge": edge,
            "ev": ev,
            "confidence": confidence,
            "decision": decision,
            "stake": stake

        })


    return results
