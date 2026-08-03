"""
Análise segmentada centralizada do Framework de Avaliação Quantitativa.

Reexpõe as segmentações já existentes em `src.backtest.historical.segments`
(competição, mercado, faixa de odds, faixa de Edge, faixa de EV, favorito
vs. underdog, casa vs. fora) e acrescenta as duas segmentações pedidas que
ainda não existiam: por MÊS e por FAIXA DE CONFIANÇA.

"Confiança" é interpretada como a probabilidade prevista pelo modelo para
a seleção apostada (coluna `probability` de `EvaluatedBet`) — é o único
valor de confiança por aposta já produzido pelo motor nos dados de
backtest, e é o mesmo valor usado em `statistics.calibration_curve`. Não
introduz nenhum conceito novo de confiança nem toca no motor de decisão.

Nenhuma fórmula de métrica é recalculada aqui: cada segmento aplica
`metrics.financial_summary` (via `segment_by_column`/`segment_by_bins`, já
existentes) a cada grupo.
"""

from typing import Dict, Sequence

import pandas as pd

from src.backtest.historical.segments import (
    DEFAULT_EDGE_BINS_PCT,
    DEFAULT_EDGE_LABELS,
    DEFAULT_EV_BINS_PCT,
    DEFAULT_EV_LABELS,
    DEFAULT_ODD_BINS,
    DEFAULT_ODD_LABELS,
    segment_by_bins,
    segment_by_column,
    segment_by_edge_range,
    segment_by_ev_range,
    segment_by_favorite_vs_underdog,
    segment_by_home_away,
    segment_by_odd_range,
)
from src.backtest.historical.segments import all_segments as _historical_all_segments

__all__ = [
    "DEFAULT_EDGE_BINS_PCT",
    "DEFAULT_EDGE_LABELS",
    "DEFAULT_EV_BINS_PCT",
    "DEFAULT_EV_LABELS",
    "DEFAULT_ODD_BINS",
    "DEFAULT_ODD_LABELS",
    "DEFAULT_CONFIDENCE_BINS_PCT",
    "DEFAULT_CONFIDENCE_LABELS",
    "segment_by_bins",
    "segment_by_column",
    "segment_by_edge_range",
    "segment_by_ev_range",
    "segment_by_favorite_vs_underdog",
    "segment_by_home_away",
    "segment_by_odd_range",
    "segment_by_confidence_range",
    "segment_by_month",
    "all_segments",
]

# Em pontos percentuais de probabilidade (a coluna `probability` guarda-a como fração).
DEFAULT_CONFIDENCE_BINS_PCT = [0.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
DEFAULT_CONFIDENCE_LABELS = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]


def segment_by_confidence_range(
    df: pd.DataFrame,
    bins: Sequence[float] = DEFAULT_CONFIDENCE_BINS_PCT,
    labels: Sequence[str] = DEFAULT_CONFIDENCE_LABELS,
) -> pd.DataFrame:
    """
    Agrupa por faixa de confiança do modelo (probabilidade prevista para a
    seleção apostada, `probability`, convertida para pontos percentuais).
    """
    return segment_by_bins(df, "probability", bins=bins, labels=labels, scale=100.0)


def segment_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por mês do calendário (`date` truncada a "AAAA-MM") e devolve
    uma linha de métricas por mês, ordenada cronologicamente.
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    working["_month"] = pd.to_datetime(working["date"], errors="coerce").dt.to_period("M").astype(str)
    working = working[working["_month"] != "NaT"]
    if working.empty:
        return pd.DataFrame()

    result = segment_by_column(working, "_month").rename(columns={"_month": "month"})
    if result.empty:
        return result
    return result.sort_values("month").reset_index(drop=True)


def all_segments(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Executa TODAS as análises por segmento pedidas pelo Framework de
    Avaliação — competição, mercado, faixa de odds, faixa de Edge, faixa
    de confiança e mês — mais os segmentos adicionais já existentes no
    Backtesting Framework (EV, favorito/underdog, casa/fora). Segmentos
    sem dados disponíveis são omitidos.
    """
    segments = dict(_historical_all_segments(df))

    month_segment = segment_by_month(df)
    if not month_segment.empty:
        segments["by_month"] = month_segment

    confidence_segment = segment_by_confidence_range(df)
    if not confidence_segment.empty:
        segments["by_confidence_range"] = confidence_segment

    return segments
