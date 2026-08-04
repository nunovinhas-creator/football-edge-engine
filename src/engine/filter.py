def is_valid_bet(result):

    # Bug: comparava com o literal "VALUE BET", que make_decision()
    # (src/engine/decision.py) nunca produz — devolve "BET 🔥"/"WAIT ⚠️"/
    # "PASS ❄️". Isto fazia com que nenhum resultado passasse nunca neste
    # filtro. Os limiares abaixo (edge, ev, confidence, odd) mantêm-se
    # exatamente iguais.
    if "BET" not in result["decision"]:
        return False

    if result["edge"] < 5:
        return False

    if result["ev"] < 10:
        return False

    if result["confidence"] == "LOW":
        return False

    if result["odd"] < 1.70:
        return False

    return True
