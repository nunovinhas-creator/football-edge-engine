from src.models.match import Match
from src.engine.bet_engine import evaluate_match


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


result = evaluate_match(game)


print("----------------")
print(result["match"])
print("Decision:", result["decision"])
print("Stake:", result["stake"], "%")
print("")
print("Motivos:")

for reason in result["reasons"]:
    print("-", reason)
