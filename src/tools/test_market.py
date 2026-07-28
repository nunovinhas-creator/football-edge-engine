from src.engine.market import analyze_market


odds = {
    "HOME": 1.65,
    "DRAW": 4.20,
    "AWAY": 4.60
}


result = analyze_market(
    odds,
    45
)


for key, value in result.items():

    print("----------------")
    print(key)
    print(value)
