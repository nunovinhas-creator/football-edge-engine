from src.models.match import Match


game = Match(
    "Benfica",
    "Porto",
    2.10,
    55,
    league="Liga Portugal",
    xg_home=1.8,
    xg_away=1.1,
    confidence=82
)


print(game.to_dict())
