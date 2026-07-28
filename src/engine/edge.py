def implied_probability(odd):

    if odd <= 0:
        return 0

    return round(
        (1 / odd) * 100,
        2
    )


def calculate_edge(
    model_probability,
    market_probability
):

    return round(
        model_probability - market_probability,
        2
    )


def calculate_ev(
    probability,
    odd
):

    return round(
        (probability / 100 * odd - 1) * 100,
        2
    )
