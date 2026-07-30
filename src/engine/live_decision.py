from dataclasses import dataclass


@dataclass
class LiveBetDecision:
    market: str
    probability: float
    odd: float
    edge: float
    action: str


def evaluate_live_market(
    probability_pct: float,
    bookie_odd: float,
    market="NEXT GOAL"
):

    implied = (1 / bookie_odd) * 100

    edge = probability_pct - implied


    if edge >= 10:
        action = "🔥 BET VALUE"

    elif edge >= 3:
        action = "⚠️ WATCH"

    else:
        action = "❄️ PASS"


    return LiveBetDecision(
        market=market,
        probability=round(probability_pct,2),
        odd=bookie_odd,
        edge=round(edge,2),
        action=action
    )
