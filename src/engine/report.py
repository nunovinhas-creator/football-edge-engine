def generate_report(result):

    print("========================")
    print("FOOTBALL EDGE ENGINE")
    print("========================")

    print()
    print(result["match"])

    if result.get("league"):
        print("Liga:", result["league"])

    print()

    print("Odd:", result["odd"])
    print(
        "Probabilidade modelo:",
        str(result["model_probability"]) + "%"
    )

    print(
        "Probabilidade mercado:",
        str(result["market_probability"]) + "%"
    )

    print()

    print(
        "Edge:",
        "+" + str(result["edge"]) + "%"
    )

    print(
        "EV:",
        "+" + str(result["ev"]) + "%"
    )

    print(
        "Confiança:",
        result["confidence"]
    )

    print(
        "Stake:",
        str(result["stake"]) + "%"
    )

    print()

    print("Decisão:", result["decision"])

    print()
    print("Motivos:")

    for reason in result["reasons"]:
        print("-", reason)

    print("========================")
