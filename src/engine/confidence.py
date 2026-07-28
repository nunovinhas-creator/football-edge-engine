def confidence_level(confidence):

    if confidence >= 80:
        return "HIGH"

    if confidence >= 60:
        return "MEDIUM"

    return "LOW"
