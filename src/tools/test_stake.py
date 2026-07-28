from src.engine.stake import calculate_stake


tests = [
    {
        "edge": 7.38,
        "confidence": "HIGH"
    },
    {
        "edge": 5.37,
        "confidence": "MEDIUM"
    },
    {
        "edge": 4.29,
        "confidence": "LOW"
    }
]


for test in tests:

    stake = calculate_stake(
        test["edge"],
        test["confidence"]
    )

    print("----------------")
    print("Edge:", test["edge"])
    print("Confidence:", test["confidence"])
    print("Stake:", stake, "%")
