from src.engine.analyzer import analyze_bet
from src.engine.live_decision import evaluate_live_market


def evaluate_hybrid(
    odd,
    pre_match_probability,
    live_probability=None,
    market="HOME"
):

    pre = analyze_bet(
        odd,
        pre_match_probability
    )

    result = {
        "market": market,
        "pre_match": pre,
        "mode": "PRE_MATCH",
        "decision": pre["decision"],
        "edge": pre["edge"]
    }


    if live_probability is not None:

        live = evaluate_live_market(
            live_probability,
            odd,
            market
        )

        result["live"] = {
            "probability": live.probability,
            "edge": live.edge,
            "action": live.action
        }

        if "BET" in live.action:
            result["mode"] = "LIVE"
            result["decision"] = live.action
            result["edge"] = live.edge

    return result
