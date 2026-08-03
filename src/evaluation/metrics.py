"""
Métricas centralizadas do Framework de Avaliação Quantitativa.

Este módulo NÃO recalcula nenhuma fórmula matemática nova: importa e
reexpõe as implementações oficiais já existentes e testadas em
`src.backtest.historical.metrics` (desempenho financeiro) e
`src.backtest.historical.statistics` (qualidade probabilística), e
acrescenta apenas pequenos wrappers de conveniência (`avg_odd`, `n_bets`,
`avg_ev_pct`, `full_summary`) que combinam essas peças num único ponto de
entrada. Nenhum modelo de previsão (Dixon-Coles, Monte Carlo, λ, Kelly,
Edge, Goal Engine, Machine Learning, Decision Engine) é tocado.

Convenções (herdadas de `src.backtest.historical`):
    - `edge`, `ev`, `kelly`, `probability` são frações (0.0-1.0); as
      médias devolvidas em "_pct" já vêm multiplicadas por 100.
    - ROI é ponderado pelo stake total; Yield é a média das rentabilidades
      individuais por aposta (ver `src.backtest.historical.metrics` para
      a distinção completa).
    - Métricas financeiras (ROI, Yield, Profit, Hit Rate, Odd média, Stake
      total, Número de apostas) devem ser calculadas sobre as apostas
      COLOCADAS (`placed_bets` / `df["placed"]`).
    - Métricas estatísticas (Brier Score, Log Loss, ECE) devem ser
      calculadas sobre TODAS as apostas avaliadas (`all_bets`), pois
      medem a qualidade do modelo, não a rentabilidade da estratégia.
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.backtest.historical.metrics import (
    equity_curve,
    expectancy_per_bet,
    hit_rate,
    max_drawdown,
    net_profit,
    profit_factor,
    roi,
    summary_metrics as financial_summary,
    total_staked,
    yield_pct,
)
from src.backtest.historical.statistics import (
    brier_score,
    calibration_curve,
    calibration_error,
    edge_distribution,
    ev_distribution,
    log_loss,
    probability_distribution,
    statistical_summary,
)

__all__ = [
    # financeiras (reexportadas)
    "equity_curve",
    "expectancy_per_bet",
    "hit_rate",
    "max_drawdown",
    "net_profit",
    "profit_factor",
    "roi",
    "financial_summary",
    "total_staked",
    "yield_pct",
    # estatísticas (reexportadas)
    "brier_score",
    "calibration_curve",
    "calibration_error",
    "edge_distribution",
    "ev_distribution",
    "log_loss",
    "probability_distribution",
    "statistical_summary",
    # wrappers de conveniência (novos, sem fórmula própria)
    "avg_odd",
    "avg_ev_pct",
    "avg_edge_pct",
    "n_bets",
    "full_summary",
]


def avg_odd(df: pd.DataFrame) -> float:
    """Odd média das apostas do subconjunto indicado."""
    if df.empty:
        return 0.0
    return round(float(df["odd"].mean()), 4)


def avg_ev_pct(df: pd.DataFrame) -> float:
    """Expected Value médio (%), fração `ev` convertida para pontos percentuais."""
    if df.empty:
        return 0.0
    return round(float(df["ev"].mean()) * 100, 2)


def avg_edge_pct(df: pd.DataFrame) -> float:
    """Edge médio (%), fração `edge` convertida para pontos percentuais."""
    if df.empty:
        return 0.0
    return round(float(df["edge"].mean()) * 100, 2)


def n_bets(df: pd.DataFrame) -> int:
    """Número de apostas no subconjunto indicado."""
    return int(len(df))


def full_summary(
    placed_df: pd.DataFrame,
    all_df: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Agrega, num único dicionário, todas as métricas pedidas pelo Framework
    de Avaliação: ROI, Yield, Profit, Hit Rate, Odd média, Stake total,
    Número de apostas, Expected Value médio (financeiras, calculadas sobre
    `placed_df`) e Brier Score, Log Loss, Calibration Error/ECE
    (estatísticas, calculadas sobre `all_df` — ou sobre `placed_df` se
    `all_df` não for fornecido).

    Não recalcula nada: delega em `financial_summary` (ROI/Yield/Profit/
    Hit Rate/drawdown/profit factor/...) e `statistical_summary`
    (Brier/LogLoss/ECE), ambas de `src.backtest.historical`.
    """
    stats_df = all_df if all_df is not None else placed_df
    return {
        **financial_summary(placed_df),
        **statistical_summary(stats_df, n_bins=n_bins),
    }
