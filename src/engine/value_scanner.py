from src.engine.ranking import rank_bets
from src.engine.live_decision import evaluate_live_market


def scan_value_opportunities(
    matches=None,
    live_probability=None,
    live_odd=None
):

    results = []

    # Pré-jogo
    if matches:

        pregame = rank_bets(matches)

        for bet in pregame:
            results.append({
                "type": "PRE_MATCH",
                "match": bet["match"],
                "edge": bet["edge"],
                "ev": bet["ev"],
                "decision": bet["decision"]
            })


    # Live
    if live_probability is not None and live_odd is not None:

        live = evaluate_live_market(
            live_probability,
            live_odd
        )

        results.append({
            "type": "LIVE",
            "market": live.market,
            "edge": live.edge,
            "decision": live.action
        })


    return sorted(
        results,
        key=lambda x: x.get("edge", 0),
        reverse=True
    )
