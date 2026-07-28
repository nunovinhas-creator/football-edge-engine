def kelly_fraction(probability, odd):
    """
    Kelly Criterion

    probability: probabilidade do modelo (0-1)
    odd: odd decimal

    retorna fração da banca
    """

    b = odd - 1
    p = probability
    q = 1 - p

    kelly = ((b * p) - q) / b

    return max(kelly, 0)


def fractional_kelly(probability, odd, fraction=0.25):
    """
    Kelly reduzido para controlar risco
    """

    return kelly_fraction(probability, odd) * fraction
