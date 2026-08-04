from dataclasses import dataclass
from typing import Optional

from src.engine.edge import calculate_edge
from src.engine.kelly import calculate_adaptive_kelly_fraction
from src.engine.kelly import kelly_fraction as _kelly_fraction

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

    def evaluate_bet(
        self,
        market: str,
        model_prob_pct: float,
        bookie_odd: float,
        lambda_tier: Optional[str] = None,
        effective_sample_size: Optional[float] = None,
    ) -> BetRecommendation:
        """
        `lambda_tier`/`effective_sample_size` são opcionais (Melhoria #6 da
        auditoria matemática — `LambdaEstimate.tier`/`.effective_sample_size`
        já produzidos por `src.engine.lambda_estimator`): quando fornecidos,
        escalam `max_kelly_fraction` pela confiança do modelo. Omissos, o
        resultado é exatamente igual ao de antes desta melhoria. Não afeta
        `edge` nem o critério BET/PASS (`min_edge`) — só o tamanho do stake
        de uma aposta já decidida.
        """
        p = model_prob_pct / 100.0
        q = 1.0 - p
        b = bookie_odd - 1.0

        if b <= 0 or p <= 0 or p > 1.0:
            return BetRecommendation(market, model_prob_pct, bookie_odd, 0.0, 0.0, "PASS")

        # Edge (%) — usa a implementação oficial e única de Edge (src/engine/edge.py)
        edge = calculate_edge(p, bookie_odd) * 100.0

        # Kelly Criterion: f* = (b*p - q) / b — reutiliza a implementação
        # oficial e única (src/engine/kelly.py), não recalcula a fórmula.
        full_kelly = _kelly_fraction(p, bookie_odd)

        # Aplica Fractional Kelly (escalada pela confiança, se disponível — Melhoria #6)
        # e limites de segurança
        if full_kelly > 0 and edge >= self.min_edge:
            adaptive_fraction = calculate_adaptive_kelly_fraction(
                self.max_kelly_fraction, lambda_tier, effective_sample_size
            )
            suggested_stake = min(full_kelly * adaptive_fraction * 100.0, 5.0)  # teto máximo de 5% da banca
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




def make_decision(edge: float, ev: float):
    """
    Compatibilidade com testes e módulos antigos.

    edge:
        Edge percentual

    ev:
        Expected Value percentual
    """

    if edge >= 5 and ev > 0:
        return "BET 🔥"

    if edge > 0 and ev > 0:
        return "WAIT ⚠️"

    return "PASS ❄️"


def evaluate_decision(edge: float, ev: float):
    """
    Compatibilidade com versões antigas do engine.
    Recebe edge e EV e devolve decisão simples.
    """

    if edge >= 5 and ev > 0:
        return "BET 🔥"

    if edge > 0:
        return "WAIT ⚠️"

    return "PASS ❄️"


def make_decision(edge: float, ev: float, bookie_odd=None):
    """
    Compatibilidade com testes antigos.
    """

    return evaluate_decision(edge, ev)
