def is_valid_bet(result):

    if result["decision"] != "VALUE BET":
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
