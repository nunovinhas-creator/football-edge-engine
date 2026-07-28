from src.engine.analyzer import analyze_bet
from src.engine.kelly import fractional_kelly
from src.engine.confidence import confidence_level
from src.engine.explanation import generate_explanation


def evaluate_match(match):

    analysis = analyze_bet(
        match.odd,
        match.probability
    )

    stake = fractional_kelly(
        match.probability / 100,
        match.odd
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
