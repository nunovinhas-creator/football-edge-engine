def predict_probability(event):

    probability = 50


    h2h = event.get(
        "head_to_head"
    ) or {}


    total_matches = h2h.get(
        "total_matches",
        0
    )


    home_rate = h2h.get(
        "home_win_rate",
        0
    )


    away_rate = h2h.get(
        "away_win_rate",
        0
    )


    avg_goals = h2h.get(
        "avg_total_goals",
        2
    )


    # vantagem de jogar em casa
    probability += 3


    # peso do histórico
    if total_matches >= 10:
        weight = 5

    elif total_matches >= 5:
        weight = 3

    elif total_matches >= 3:
        weight = 2

    else:
        weight = 1


    # comparação histórica
    if home_rate > away_rate:

        probability += weight


    elif away_rate > home_rate:

        probability -= weight



    # perfil de golos
    if avg_goals >= 3:

        probability += 2


    elif avg_goals < 2:

        probability -= 2



    # amostra pequena reduz confiança
    if total_matches < 3:

        probability = (
            probability + 50
        ) / 2



    # limites realistas

    if probability > 65:
        probability = 65


    if probability < 35:
        probability = 35


    return round(
        probability,
        2
    )
