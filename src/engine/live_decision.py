from dataclasses import dataclass

from src.engine.edge import calculate_edge


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

    p = probability_pct / 100.0

    if bookie_odd <= 1.0 or p <= 0.0 or p > 1.0:
        return LiveBetDecision(
            market=market,
            probability=round(probability_pct, 2),
            odd=bookie_odd,
            edge=0.0,
            action="❄️ PASS"
        )

    # Edge (%) — usa a implementação oficial e única de Edge (src/engine/edge.py)
    edge = calculate_edge(p, bookie_odd) * 100.0


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
