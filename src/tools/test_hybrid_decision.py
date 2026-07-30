from src.engine.decision import DecisionEngine
from src.engine.live_decision import evaluate_live_market


print("====================")
print("PRE-MATCH")
print("====================")

engine = DecisionEngine()

pre = engine.evaluate_bet(
    market="HOME",
    model_prob_pct=55,
    bookie_odd=2.10
)

print(pre)


print("====================")
print("LIVE")
print("====================")

live = evaluate_live_market(
    probability_pct=72,
    bookie_odd=2.10,
    market="NEXT GOAL"
)

print(live)

print("====================")
