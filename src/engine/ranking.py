from src.engine.filter import is_valid_bet
from src.engine.analyzer import analyze_bet
from src.engine.live_decision import evaluate_live_market


def rank_bets(matches):

    results = []

    for match in matches:

        analysis = analyze_bet(
            match.odds,
            match.probability
        )

        result = {
            "match": f"{match.home} vs {match.away}",
            "edge": analysis["edge"],
            "ev": analysis.get("ev", 0),
            "decision": analysis.get("decision", "PASS"),
            "stake": analysis.get("stake", 0),
            "mode": "PRE_MATCH"
        }

        # integração live quando existir estado live
        if hasattr(match, "live_probability") and hasattr(match, "live_odd"):

            live = evaluate_live_market(
                match.live_probability,
                match.live_odd
            )

            result["live"] = {
                "market": live.market,
                "probability": live.probability,
                "odd": live.odd,
                "edge": live.edge,
                "action": live.action
            }

            if "BET" in live.action:
                result["decision"] = "LIVE BET"

            elif "WATCH" in live.action:
                result["decision"] = "LIVE WATCH"

            else:
                result["decision"] = "PASS"


        results.append(result)


    return sorted(
        results,
        key=lambda x: x["edge"],
        reverse=True
    )



def create_ranking(results):

    value_bets = []
    watchlist = []

    for result in results:

        if is_valid_bet(result):
            value_bets.append(result)

        elif (
            result.get("decision") in [
                "WATCH",
                "LIVE WATCH"
            ]
            or result.get("ev", 0) > 0
        ):
            watchlist.append(result)


    return {
        "value_bets": sorted(
            value_bets,
            key=lambda x: x["edge"],
            reverse=True
        ),
        "watchlist": sorted(
            watchlist,
            key=lambda x: x.get("ev",0),
            reverse=True
        )
    }
