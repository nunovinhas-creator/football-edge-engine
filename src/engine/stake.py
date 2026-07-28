def calculate_stake(
    edge,
    confidence,
    h2h_matches=0
):

    stake = 0


    if confidence == "HIGH":

        if h2h_matches >= 5:
            stake = edge * 0.5

        else:
            stake = edge * 0.25


    elif confidence == "MEDIUM":

        if h2h_matches >= 3:
            stake = edge * 0.25

        else:
            stake = edge * 0.15


    else:

        stake = edge * 0.05


    if stake > 5:
        stake = 5


    if stake < 0:
        stake = 0


    return round(
        stake,
        2
    )
