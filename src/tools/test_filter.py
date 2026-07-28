from src.engine.filter import is_valid_bet


tests = [

    {
        "decision": "VALUE BET",
        "edge": 7.38,
        "ev": 15.5,
        "confidence": "HIGH",
        "odd": 2.10
    },

    {
        "decision": "VALUE BET",
        "edge": 28.26,
        "ev": 130,
        "confidence": "LOW",
        "odd": 4.60
    },

    {
        "decision": "WATCH",
        "edge": 4.29,
        "ev": 12,
        "confidence": "MEDIUM",
        "odd": 2.80
    }

]


for test in tests:

    result = is_valid_bet(test)

    print("----------------")
    print(test)
    print("VALID:", result)
