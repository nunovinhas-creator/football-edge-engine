"""
Análises por segmento do Backtesting Framework.

Reparte o mesmo conjunto de apostas avaliadas por diferentes dimensões
(competição, mercado, intervalo de odds, intervalo de Edge, intervalo de
EV, favorito vs underdog, casa vs fora) e aplica `metrics.summary_metrics`
a cada segmento — sem recalcular nenhuma fórmula de Edge/EV/Kelly.
"""

from typing import Dict, List, Optional, Sequence

import pandas as pd

from .metrics import summary_metrics

DEFAULT_ODD_BINS = [1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]
DEFAULT_ODD_LABELS = ["1.00-1.50", "1.50-2.00", "2.00-3.00", "3.00-5.00", "5.00+"]

# Em pontos percentuais de Edge/EV (o DataFrame guarda edge/ev como fração).
DEFAULT_EDGE_BINS_PCT = [-100.0, 0.0, 3.0, 5.0, 7.0, 10.0, 15.0, 100.0]
DEFAULT_EDGE_LABELS = ["<0%", "0-3%", "3-5%", "5-7%", "7-10%", "10-15%", "15%+"]

DEFAULT_EV_BINS_PCT = DEFAULT_EDGE_BINS_PCT
DEFAULT_EV_LABELS = DEFAULT_EDGE_LABELS


def segment_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Agrupa por uma coluna categórica (ex. `competition`, `market`,
    `home_or_away`) e devolve uma linha de métricas por valor único,
    ordenada por número de apostas (decrescente).
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    rows = []
    for value, group in df.groupby(column, dropna=False):
        metrics = summary_metrics(group)
        metrics[column] = value
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    ordered_columns = [column] + [c for c in result.columns if c != column]
    return result[ordered_columns].sort_values("n_bets", ascending=False).reset_index(drop=True)


def segment_by_bins(
    df: pd.DataFrame,
    column: str,
    bins: Sequence[float],
    labels: Sequence[str],
    scale: float = 1.0,
) -> pd.DataFrame:
    """
    Agrupa por intervalos numéricos de uma coluna (ex. odd, edge, ev).

    `scale` permite converter a coluna original (fração) para a mesma
    unidade dos `bins` (ex. scale=100 para converter edge/ev fracionário
    em pontos percentuais antes de aplicar os limites).
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    working["_bucket"] = pd.cut(working[column] * scale, bins=bins, labels=labels, right=False)

    rows = []
    for label in labels:
        group = working[working["_bucket"] == label]
        if group.empty:
            continue
        metrics = summary_metrics(group)
        metrics["range"] = label
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    ordered_columns = ["range"] + [c for c in result.columns if c != "range"]
    return result[ordered_columns]


def segment_by_odd_range(
    df: pd.DataFrame, bins: Sequence[float] = DEFAULT_ODD_BINS, labels: Sequence[str] = DEFAULT_ODD_LABELS
) -> pd.DataFrame:
    return segment_by_bins(df, "odd", bins=bins, labels=labels, scale=1.0)


def segment_by_edge_range(
    df: pd.DataFrame,
    bins: Sequence[float] = DEFAULT_EDGE_BINS_PCT,
    labels: Sequence[str] = DEFAULT_EDGE_LABELS,
) -> pd.DataFrame:
    return segment_by_bins(df, "edge", bins=bins, labels=labels, scale=100.0)


def segment_by_ev_range(
    df: pd.DataFrame, bins: Sequence[float] = DEFAULT_EV_BINS_PCT, labels: Sequence[str] = DEFAULT_EV_LABELS
) -> pd.DataFrame:
    return segment_by_bins(df, "ev", bins=bins, labels=labels, scale=100.0)


def segment_by_favorite_vs_underdog(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_favorite" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working["_label"] = working["is_favorite"].map({True: "FAVORITE", False: "UNDERDOG"})
    return segment_by_column(working, "_label").rename(columns={"_label": "selection_type"})


def segment_by_home_away(df: pd.DataFrame) -> pd.DataFrame:
    return segment_by_column(df, "home_or_away")


def all_segments(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Executa todas as análises por segmento definidas nos requisitos e
    devolve um dicionário {nome_do_segmento: DataFrame de métricas}.
    Segmentos sem dados disponíveis (coluna ausente/vazia) são omitidos.
    """
    segments = {
        "by_competition": segment_by_column(df, "competition"),
        "by_market": segment_by_column(df, "market"),
        "by_odd_range": segment_by_odd_range(df),
        "by_edge_range": segment_by_edge_range(df),
        "by_ev_range": segment_by_ev_range(df),
        "by_favorite_vs_underdog": segment_by_favorite_vs_underdog(df),
        "by_home_away": segment_by_home_away(df),
    }
    return {name: result for name, result in segments.items() if not result.empty}
