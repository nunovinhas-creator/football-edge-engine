from src.collector.client import get_matches
from src.engine.ranking import rank_bets
from src.engine.report import generate_report


matches = get_matches()


ranking = rank_bets(matches)


print("\n🔥 VALUE BETS\n")

for bet in ranking["value_bets"]:
    generate_report(bet)


print("\n👀 WATCHLIST\n")

for bet in ranking["watchlist"]:
    generate_report(bet)
