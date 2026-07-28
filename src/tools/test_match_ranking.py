from src.models.match import Match
from src.engine.ranking import rank_bets


matches = [

    Match(
        "Benfica",
        "Porto",
        2.10,
        55
    ),

    Match(
        "Milan",
        "Roma",
        1.90,
        58
    ),

    Match(
        "River",
        "Boca",
        2.80,
        40
    )

]


ranking = rank_bets(matches)


for bet in ranking:

    print("----------------")
    print(bet["match"])
    print("Edge:", bet["edge"])
    print("EV:", bet["ev"])
    print("Decision:", bet["decision"])
    print("Stake:", bet["stake"], "%")
