from src.engine.analyzer import analyze_bet


result = analyze_bet(
    odd=2.10,
    model_probability=55
)


print("----------------")
print("ANÁLISE")
print("----------------")

for key, value in result.items():
    print(f"{key}: {value}")
