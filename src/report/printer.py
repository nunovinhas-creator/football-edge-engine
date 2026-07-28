def print_report(result):

    print("========================")
    print("FOOTBALL EDGE ENGINE")
    print("========================")
    print()

    print(result["match"])
    print()

    print("Mercado:", result.get("market"))
    print("Odd:", result["odd"])
    print()

    print(
        "Probabilidade modelo:",
        result["model_probability"],
        "%"
    )

    print(
        "Probabilidade mercado:",
        result["market_probability"],
        "%"
    )

    print()

    print(
        "Edge:",
        "+" if result["edge"] > 0 else "",
        result["edge"],
        "%"
    )

    print(
        "EV:",
        "+" if result["ev"] > 0 else "",
        result["ev"],
        "%"
    )

    print(
        "Confiança:",
        result["confidence"]
    )

    print(
        "Motivo confiança:",
        result.get(
            "confidence_reason",
            "N/A"
        )
    )

    print(
        "H2H analisados:",
        result.get(
            "h2h_matches",
            0
        ),
        "jogos"
    )

    print(
        "Stake:",
        result["stake"],
        "%"
    )

    print()

    print(
        "Decisão:",
        result["decision"]
    )

    print()

    print("Motivos:")

    for reason in result.get("reasons", []):
        print("-", reason)

    print("========================")
