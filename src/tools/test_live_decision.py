from src.engine.live_decision import evaluate_live_market


result = evaluate_live_market(
    probability_pct=72,
    bookie_odd=2.10,
    market="NEXT GOAL"
)


print("====================")
print(result)
print("====================")
