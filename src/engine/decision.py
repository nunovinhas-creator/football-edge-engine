def make_decision(edge, ev):

    if edge >= 5 and ev >= 10:
        return "VALUE BET"

    if ev >= 5:
        return "WATCH"

    return "NO BET"
