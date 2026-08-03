"""
Threshold Analysis do Backtesting Framework.

Avalia, para diferentes limiares mínimos de Edge (ou EV), qual seria o
desempenho da estratégia SE apenas as apostas com Edge/EV acima do limiar
tivessem sido colocadas — independentemente da decisão histórica real do
motor. O objetivo é descobrir empiricamente os melhores thresholds, não
alterar a decisão do motor em produção.
"""

from typing import Optional, Sequence

import pandas as pd

from .metrics import hit_rate, net_profit, roi, yield_pct

DEFAULT_EDGE_THRESHOLDS_PCT = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
DEFAULT_EV_THRESHOLDS_PCT = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]


def _threshold_table(df: pd.DataFrame, column: str, thresholds_pct: Sequence[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds_pct:
        subset = df[df[column] * 100 >= threshold]
        rows.append(
            {
                "threshold_pct": threshold,
                "n_bets": int(len(subset)),
                "hit_rate_pct": hit_rate(subset),
                "roi_pct": roi(subset),
                "yield_pct": yield_pct(subset),
                "profit": net_profit(subset),
            }
        )
    return pd.DataFrame(rows)


def edge_threshold_analysis(
    df: pd.DataFrame, thresholds_pct: Sequence[float] = DEFAULT_EDGE_THRESHOLDS_PCT
) -> pd.DataFrame:
    """
    Para cada limiar de Edge (>=, em pontos percentuais) devolve ROI,
    Yield, número de apostas, lucro e taxa de acerto.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["threshold_pct", "n_bets", "hit_rate_pct", "roi_pct", "yield_pct", "profit"]
        )
    return _threshold_table(df, "edge", thresholds_pct)


def ev_threshold_analysis(
    df: pd.DataFrame, thresholds_pct: Sequence[float] = DEFAULT_EV_THRESHOLDS_PCT
) -> pd.DataFrame:
    """Equivalente a `edge_threshold_analysis`, mas usando o EV como filtro."""
    if df.empty:
        return pd.DataFrame(
            columns=["threshold_pct", "n_bets", "hit_rate_pct", "roi_pct", "yield_pct", "profit"]
        )
    return _threshold_table(df, "ev", thresholds_pct)


def best_threshold(threshold_table: pd.DataFrame, by: str = "roi_pct", min_bets: int = 1) -> Optional[dict]:
    """
    Devolve a linha do `threshold_table` com melhor valor da coluna `by`
    (por omissão, ROI), entre os limiares com pelo menos `min_bets`
    apostas — evita escolher um limiar "vencedor" suportado por 1-2 apostas.
    """
    if threshold_table.empty:
        return None
    candidates = threshold_table[threshold_table["n_bets"] >= min_bets]
    if candidates.empty:
        return None
    best_row = candidates.loc[candidates[by].idxmax()]
    return best_row.to_dict()
