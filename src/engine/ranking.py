from src.engine.filter import is_valid_bet
from src.engine.analyzer import analyze_bet


def rank_bets(matches):

    results = []

    for match in matches:

        analysis = analyze_bet(
            match.odds,
            match.probability
        )

        results.append({
            "match": f"{match.home} vs {match.away}",
            "edge": analysis["edge"],
            "ev": analysis.get("ev", 0),
            "decision": analysis.get("decision", "PASS"),
            "stake": analysis.get("stake", 0)
        })

    results = sorted(
        results,
        key=lambda x: x["edge"],
        reverse=True
    )

    return results


def create_ranking(results):

    value_bets = []
    watchlist = []

    for result in results:

        if is_valid_bet(result):
            value_bets.append(result)

        elif (
            result.get("decision") == "WATCH"
            or result.get("ev", 0) > 0
        ):
            watchlist.append(result)

    value_bets = sorted(
        value_bets,
        key=lambda x: x["edge"],
        reverse=True
    )

    watchlist = sorted(
        watchlist,
        key=lambda x: x["ev"],
        reverse=True
    )

    return {
        "value_bets": value_bets,
        "watchlist": watchlist
    }
