"""
Avaliação por aposta do Backtesting Framework.

Para cada `HistoricalBet` calcula: probabilidade do modelo, probabilidade
de mercado, Edge, EV, Kelly, Stake e lucro líquido — reutilizando sempre as
implementações oficiais existentes:

    - src.engine.edge.implied_probability / calculate_edge / calculate_ev
    - src.engine.kelly.kelly_fraction

Nenhuma fórmula matemática é recalculada ou alterada aqui. Este módulo
apenas orquestra os dados históricos através dessas fórmulas e regista o
resultado por aposta, incluindo hipotéticos (stake/profit) que permitem
reanalisar as apostas sob critérios diferentes da decisão histórica do
motor (ver `thresholds.py` e `segments.py`).
"""

from typing import List, Optional

import pandas as pd

from src.engine.edge import calculate_edge, calculate_ev, implied_probability
from src.engine.kelly import kelly_fraction

from .models import EvaluatedBet, HistoricalBet
from .staking import FlatStake, StakingStrategy

FAVORITE_ODD_THRESHOLD = 2.0


def _infer_is_favorite(bet: HistoricalBet) -> bool:
    if bet.is_favorite is not None:
        return bet.is_favorite
    return bet.odd <= FAVORITE_ODD_THRESHOLD


def evaluate_bet(bet: HistoricalBet, staking: Optional[StakingStrategy] = None) -> EvaluatedBet:
    """
    Avalia uma única aposta histórica, calculando as métricas oficiais de
    Edge/EV/Kelly e o resultado financeiro hipotético (stake/profit),
    independentemente de a decisão histórica do motor ter sido "apostar".
    """
    staking = staking or FlatStake(unit=1.0)

    probability = bet.model_prob
    market_probability = implied_probability(bet.odd)
    edge = calculate_edge(probability, bet.odd)
    ev = calculate_ev(probability, bet.odd)
    kelly = kelly_fraction(probability, bet.odd)

    stake = staking.stake_for(probability, bet.odd)
    won = bet.won
    profit = stake * (bet.odd - 1.0) if won else -stake

    placed = HistoricalBet.is_bet_decision(bet.engine_decision)

    return EvaluatedBet(
        match=bet.match,
        date=bet.date,
        market=bet.market,
        competition=bet.competition,
        home_team=bet.home_team,
        away_team=bet.away_team,
        home_or_away=bet.home_or_away,
        is_favorite=_infer_is_favorite(bet),
        odd=bet.odd,
        probability=probability,
        market_probability=market_probability,
        edge=edge,
        ev=ev,
        kelly=kelly,
        stake=stake,
        engine_decision=bet.engine_decision,
        placed=placed,
        won=won,
        profit=round(profit, 4),
        model_confidence=bet.model_confidence,
        lambda_tier=bet.lambda_tier,
        effective_sample_size=bet.effective_sample_size,
    )


def evaluate_bets(
    bets: List[HistoricalBet], staking: Optional[StakingStrategy] = None
) -> pd.DataFrame:
    """
    Avalia uma lista de apostas históricas e devolve um DataFrame — uma
    linha por aposta, com todas as métricas calculadas por `evaluate_bet`.
    """
    staking = staking or FlatStake(unit=1.0)
    rows = [evaluate_bet(bet, staking=staking).to_dict() for bet in bets]
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df
