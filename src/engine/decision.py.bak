from dataclasses import dataclass

@dataclass
class BetRecommendation:
    market: str
    model_prob: float
    bookie_odd: float
    edge_pct: float
    kelly_stake_pct: float
    action: str  # BET, PASS, WAIT

class DecisionEngine:
    def __init__(self, max_kelly_fraction: float = 0.25, min_edge: float = 5.0):
        """
        :param max_kelly_fraction: Fractional Kelly (ex: 0.25 para Quarter Kelly, mais seguro)
        :param min_edge: Edge mínima em % para considerar valor
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_edge = min_edge

    def evaluate_bet(self, market: str, model_prob_pct: float, bookie_odd: float) -> BetRecommendation:
        p = model_prob_pct / 100.0
        q = 1.0 - p
        b = bookie_odd - 1.0

        if b <= 0 or p <= 0:
            return BetRecommendation(market, model_prob_pct, bookie_odd, 0.0, 0.0, "PASS")

        # Implied Probability da casa de apostas
        implied_prob = 1.0 / bookie_odd
        
        # Edge (%)
        edge = (p - implied_prob) * 100.0

        # Kelly Criterion: f* = (b*p - q) / b
        full_kelly = (b * p - q) / b
        
        # Aplica Fractional Kelly e limites de segurança
        if full_kelly > 0 and edge >= self.min_edge:
            suggested_stake = min(full_kelly * self.max_kelly_fraction * 100.0, 5.0)  # teto máximo de 5% da banca
            action = "BET 🔥"
        else:
            suggested_stake = 0.0
            action = "PASS ❄️"

        return BetRecommendation(
            market=market,
            model_prob=round(model_prob_pct, 1),
            bookie_odd=bookie_odd,
            edge_pct=round(edge, 1),
            kelly_stake_pct=round(suggested_stake, 2),
            action=action
        )
