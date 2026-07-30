from src.engine.analyzer import analyze_bet
from src.engine.kelly import fractional_kelly
from src.engine.confidence import confidence_level
from src.engine.explanation import generate_explanation


def evaluate_match(match):

    analysis = analyze_bet(
        getattr(match, 'odd', match.odds),
        match.probability
    )

    stake = fractional_kelly(
        match.probability / 100,
        getattr(match, "odd", match.odds)
    )

    analysis["match"] = (
        f"{match.home} vs {match.away}"
    )

    analysis["league"] = match.league

    analysis["confidence"] = (
        confidence_level(match.confidence)
        if match.confidence
        else "UNKNOWN"
    )

    analysis["xg"] = (
        f"{match.xg_home} - {match.xg_away}"
        if match.xg_home and match.xg_away
        else None
    )

    analysis["stake"] = round(
        stake * 100,
        2
    )

    analysis["reasons"] = generate_explanation(
        analysis
    )

    return analysis




def evaluate_bet(
    odd=None,
    probability=None,
    model_probability=None,
    **kwargs
):
    """
    Wrapper compatibilidade v3/v4.
    Aceita:
      - probability
      - model_probability
    """

    if model_probability is not None:
        probability = model_probability

    if odd is None:
        odd = kwargs.get("bookie_odd")

    if odd is None or probability is None:
        return {
            "error": "missing parameters"
        }

    prob = probability / 100.0

    ev = (prob * odd) - 1.0

    return {
        "odd": odd,
        "probability": probability,
        "ev": round(ev * 100, 2),
        "edge": round((prob - (1 / odd)) * 100, 2) if odd > 1 else -100,
        "value": ev > 0
    }
