from src.engine.bet_engine import evaluate_bet


result = evaluate_bet(
    odd=2.10,
    model_probability=55
)


print("----------------")
print("BET ENGINE")
print("----------------")


for key, value in result.items():
    print(f"{key}: {value}")
