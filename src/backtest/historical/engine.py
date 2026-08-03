"""
Orquestrador do Backtesting Framework histórico.

`BacktestEngine.run(...)` é o ponto de entrada único do módulo: recebe
dados históricos (lista de dicts, ou já como `HistoricalBet`), avalia cada
aposta reutilizando as fórmulas oficiais de Edge/EV/Kelly, e devolve um
`BacktestReport` com métricas globais, estatísticas, análises por
segmento e threshold analysis.

Este módulo não altera o motor de previsão em nenhuma forma — consome
apenas os seus outputs históricos (probabilidade prevista e decisão já
tomada) fornecidos nos dados de entrada.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from .evaluator import evaluate_bets
from .metrics import summary_metrics
from .models import HistoricalBet
from .report import BacktestReport
from .segments import all_segments
from .staking import FlatStake, StakingStrategy
from .statistics import calibration_curve, statistical_summary
from .thresholds import (
    DEFAULT_EDGE_THRESHOLDS_PCT,
    DEFAULT_EV_THRESHOLDS_PCT,
    edge_threshold_analysis,
    ev_threshold_analysis,
)

BetsInput = Union[
    Iterable[Dict[str, Any]],
    Iterable[HistoricalBet],
    pd.DataFrame,
]


def _normalize_bets(bets: BetsInput) -> List[HistoricalBet]:
    if isinstance(bets, pd.DataFrame):
        records = bets.to_dict(orient="records")
        return [HistoricalBet.from_dict(row) for row in records]

    bets = list(bets)
    if not bets:
        return []
    if isinstance(bets[0], HistoricalBet):
        return bets
    return [HistoricalBet.from_dict(row) for row in bets]


class BacktestEngine:
    """
    Motor de Backtesting histórico. Não substitui, recalcula nem altera o
    motor de previsão (Poisson, Dixon-Coles, Monte Carlo, Goal Engine,
    Kelly, Edge, EV) — apenas mede o desempenho passado das decisões que
    esse motor já produziu.
    """

    def __init__(
        self,
        staking: Optional[StakingStrategy] = None,
        edge_thresholds_pct: Sequence[float] = DEFAULT_EDGE_THRESHOLDS_PCT,
        ev_thresholds_pct: Sequence[float] = DEFAULT_EV_THRESHOLDS_PCT,
        calibration_bins: int = 10,
    ):
        self.staking = staking or FlatStake(unit=1.0)
        self.edge_thresholds_pct = edge_thresholds_pct
        self.ev_thresholds_pct = ev_thresholds_pct
        self.calibration_bins = calibration_bins

    def run(self, bets: BetsInput) -> BacktestReport:
        """
        Executa o backtest completo sobre os dados históricos fornecidos.

        `bets` pode ser: um DataFrame, uma lista de dicts (aceitando chaves
        em português ou inglês, ver `HistoricalBet.from_dict`), ou uma
        lista de `HistoricalBet` já construídos.
        """
        historical_bets = _normalize_bets(bets)
        all_evaluated = evaluate_bets(historical_bets, staking=self.staking)

        if all_evaluated.empty:
            placed = all_evaluated
        else:
            placed = all_evaluated[all_evaluated["placed"]].reset_index(drop=True)

        return BacktestReport(
            all_bets=all_evaluated,
            placed_bets=placed,
            global_metrics=summary_metrics(placed),
            statistical_metrics=statistical_summary(all_evaluated, n_bins=self.calibration_bins),
            calibration_curve=calibration_curve(all_evaluated, n_bins=self.calibration_bins),
            segments=all_segments(placed),
            edge_thresholds=edge_threshold_analysis(all_evaluated, self.edge_thresholds_pct),
            ev_thresholds=ev_threshold_analysis(all_evaluated, self.ev_thresholds_pct),
        )


def run_backtest(
    bets: BetsInput,
    staking: Optional[StakingStrategy] = None,
    edge_thresholds_pct: Sequence[float] = DEFAULT_EDGE_THRESHOLDS_PCT,
    ev_thresholds_pct: Sequence[float] = DEFAULT_EV_THRESHOLDS_PCT,
) -> BacktestReport:
    """Atalho funcional para `BacktestEngine(...).run(bets)`."""
    engine = BacktestEngine(
        staking=staking,
        edge_thresholds_pct=edge_thresholds_pct,
        ev_thresholds_pct=ev_thresholds_pct,
    )
    return engine.run(bets)
