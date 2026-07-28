def generate_explanation(result):

    reasons = []

    if result["edge"] >= 5:
        reasons.append(
            f"Edge elevado (+{result['edge']}%)"
        )

    if result["ev"] >= 10:
        reasons.append(
            f"EV positivo (+{result['ev']}%)"
        )

    if result["confidence"] == "HIGH":
        reasons.append(
            "Modelo com alta confiança"
        )

    if result["xg"]:
        reasons.append(
            f"xG considerado: {result['xg']}"
        )

    return reasons
