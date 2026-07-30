from src.engine.hybrid_engine import evaluate_hybrid


result = evaluate_hybrid(
    odd=2.10,
    pre_match_probability=55,
    live_probability=72,
    market="NEXT GOAL"
)

print("====================")
print(result)
print("====================")
