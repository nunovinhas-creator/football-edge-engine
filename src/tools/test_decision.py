from src.engine.decision import make_decision


tests = [
    {
        "edge": 7.38,
        "ev": 15.5
    },
    {
        "edge": 4.29,
        "ev": 12
    },
    {
        "edge": -5,
        "ev": -10
    }
]


for test in tests:

    decision = make_decision(
        test["edge"],
        test["ev"]
    )

    print("----------------")
    print("Edge:", test["edge"])
    print("EV:", test["ev"])
    print("Decision:", decision)
