from src.engine.filter import is_valid_bet


def create_ranking(results):

    value_bets = []
    watchlist = []


    for result in results:

        if is_valid_bet(result):

            value_bets.append(result)

        elif (
            result["decision"] == "WATCH"
            or result["ev"] > 0
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
