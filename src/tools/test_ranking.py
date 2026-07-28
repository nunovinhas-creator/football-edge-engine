from src.engine.ranking import create_ranking


results = [

    {
        "match": "Benfica vs Porto",
        "edge": 7.38,
        "ev": 15.5,
        "decision": "VALUE BET",
        "confidence": "HIGH",
        "odd": 2.10
    },

    {
        "match": "Milan vs Roma",
        "edge": 5.37,
        "ev": 10.2,
        "decision": "VALUE BET",
        "confidence": "MEDIUM",
        "odd": 1.90
    },

    {
        "match": "River vs Boca",
        "edge": 4.29,
        "ev": 12,
        "decision": "WATCH",
        "confidence": "MEDIUM",
        "odd": 2.80
    }

]


ranking = create_ranking(results)


print("🔥 VALUE BETS")
print()

for bet in ranking["value_bets"]:
    print(bet)


print()
print("👀 WATCHLIST")
print()

for bet in ranking["watchlist"]:
    print(bet)
