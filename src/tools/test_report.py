from src.report.printer import print_report


result = {

    "match": "Benfica vs Porto",
    "market": "HOME",
    "odd": 2.10,

    "model_probability": 55,
    "market_probability": 47.62,

    "edge": 7.38,
    "ev": 15.5,

    "confidence": "HIGH",
    "stake": 3.52,

    "decision": "VALUE BET",

    "reasons": [
        "Edge elevado (+7.38%)",
        "EV positivo (+15.5%)",
        "Modelo com alta confiança"
    ]

}


print_report(result)
