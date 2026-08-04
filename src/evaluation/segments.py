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

from typing import Any, Dict, Sequence

import pandas as pd

from src.backtest.historical.metrics import summary_metrics as financial_summary
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
from src.backtest.historical.statistics import statistical_summary

__all__ = [
    "DEFAULT_EDGE_BINS_PCT",
    "DEFAULT_EDGE_LABELS",
    "DEFAULT_EV_BINS_PCT",
    "DEFAULT_EV_LABELS",
    "DEFAULT_ODD_BINS",
    "DEFAULT_ODD_LABELS",
    "DEFAULT_CONFIDENCE_BINS_PCT",
    "DEFAULT_CONFIDENCE_LABELS",
    "DEFAULT_EFFECTIVE_SAMPLE_SIZE_BINS",
    "DEFAULT_EFFECTIVE_SAMPLE_SIZE_LABELS",
    "segment_by_bins",
    "segment_by_column",
    "segment_by_edge_range",
    "segment_by_ev_range",
    "segment_by_favorite_vs_underdog",
    "segment_by_home_away",
    "segment_by_odd_range",
    "segment_by_confidence_range",
    "segment_by_month",
    "segment_by_lambda_tier",
    "segment_by_model_confidence",
    "segment_by_effective_sample_size_range",
    "all_segments",
    "all_confidence_segments",
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


# --------------------------------------------------------------------------
# Melhoria #8 (auditoria matemática): segmentação por confiança REAL do
# modelo — `LambdaEstimate.tier` / `.effective_sample_size`, propagados
# como metadados opcionais até `HistoricalBet`/`EvaluatedBet` (ver
# `src.backtest.historical.models`), em vez de `probability` (a
# probabilidade da seleção apostada, já coberta por
# `segment_by_confidence_range` acima — um conceito diferente: quão
# provável o modelo achou a aposta, não quanta informação sustentava essa
# estimativa).
#
# Ao contrário dos segmentos acima (que recebem só `placed_bets` e usam
# `summary_metrics`, puramente financeiro), estes recebem `all_bets`
# porque precisam de Brier Score / Log Loss (medem a qualidade do modelo
# sobre TODAS as apostas avaliadas, colocadas ou não — mesma convenção de
# `evaluation.metrics.full_summary`), a par de ROI/Yield/nº de apostas
# (financeiro, só sobre o subconjunto "placed" de cada grupo). Nenhuma
# fórmula nova: reutiliza `financial_summary`/`statistical_summary`
# exatamente como o resto deste módulo.
# --------------------------------------------------------------------------

DEFAULT_EFFECTIVE_SAMPLE_SIZE_BINS = [0.0, 2.0, 4.0, 8.0, 15.0, float("inf")]
DEFAULT_EFFECTIVE_SAMPLE_SIZE_LABELS = ["0-2", "2-4", "4-8", "8-15", "15+"]


def _confidence_group_metrics(group: pd.DataFrame) -> Dict[str, Any]:
    """Financeiro (apostas colocadas do grupo) + estatístico (todas as apostas do grupo)."""
    placed_group = group[group["placed"]] if "placed" in group.columns else group.iloc[0:0]
    metrics: Dict[str, Any] = {}
    metrics.update(financial_summary(placed_group))
    metrics.update(statistical_summary(group))
    return metrics


def segment_by_lambda_tier(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta por `lambda_tier` (proveniência da estimativa de lambda —
    "recent_matches" | "h2h_goal_totals" | "avg_total_goals_or_prior").

    Apostas sem este metadado (ficheiros antigos, sem a Melhoria #8)
    são ignoradas aqui, sem erro — retrocompatibilidade.
    """
    if all_df.empty or "lambda_tier" not in all_df.columns:
        return pd.DataFrame()
    working = all_df[all_df["lambda_tier"].notna()]
    if working.empty:
        return pd.DataFrame()

    rows = []
    for tier, group in working.groupby("lambda_tier"):
        row = {"lambda_tier": tier}
        row.update(_confidence_group_metrics(group))
        rows.append(row)

    result = pd.DataFrame(rows)
    ordered_columns = ["lambda_tier"] + [c for c in result.columns if c != "lambda_tier"]
    return result[ordered_columns].sort_values("n_bets", ascending=False).reset_index(drop=True)


def segment_by_model_confidence(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta por `model_confidence` ("HIGH"/"MEDIUM"/"LOW", ver
    `src.engine.lambda_estimator.classify_model_confidence`).

    Apostas sem este metadado são ignoradas aqui, sem erro —
    retrocompatibilidade.
    """
    if all_df.empty or "model_confidence" not in all_df.columns:
        return pd.DataFrame()
    working = all_df[all_df["model_confidence"].notna()]
    if working.empty:
        return pd.DataFrame()

    rows = []
    for label, group in working.groupby("model_confidence"):
        row = {"model_confidence": label}
        row.update(_confidence_group_metrics(group))
        rows.append(row)

    result = pd.DataFrame(rows)
    ordered_columns = ["model_confidence"] + [c for c in result.columns if c != "model_confidence"]
    return result[ordered_columns].sort_values("n_bets", ascending=False).reset_index(drop=True)


def segment_by_effective_sample_size_range(
    all_df: pd.DataFrame,
    bins: Sequence[float] = DEFAULT_EFFECTIVE_SAMPLE_SIZE_BINS,
    labels: Sequence[str] = DEFAULT_EFFECTIVE_SAMPLE_SIZE_LABELS,
) -> pd.DataFrame:
    """
    Segmenta por faixa de `effective_sample_size` (dimensão de amostra
    efetiva por detrás da estimativa de lambda — ver
    `src.engine.lambda_estimator.LambdaEstimate.effective_sample_size`).

    Apostas sem este metadado são ignoradas aqui, sem erro —
    retrocompatibilidade.
    """
    if all_df.empty or "effective_sample_size" not in all_df.columns:
        return pd.DataFrame()
    working = all_df[all_df["effective_sample_size"].notna()].copy()
    if working.empty:
        return pd.DataFrame()
    working["_bucket"] = pd.cut(
        working["effective_sample_size"], bins=bins, labels=labels, right=False
    )

    rows = []
    for label in labels:
        group = working[working["_bucket"] == label]
        if group.empty:
            continue
        row = {"range": label}
        row.update(_confidence_group_metrics(group))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    ordered_columns = ["range"] + [c for c in result.columns if c != "range"]
    return result[ordered_columns]


def all_confidence_segments(all_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Todos os segmentos de confiança REAL do modelo pedidos pela Melhoria
    #8: `by_lambda_tier`, `by_effective_sample_size_range`,
    `by_model_confidence`. Cada tabela combina ROI/Yield/Nº de apostas
    (financeiro, sobre as apostas colocadas de cada grupo) com Brier
    Score/Log Loss (estatístico, sobre todas as apostas avaliadas desse
    grupo). Recebe `all_bets` (não `placed_bets`), ao contrário de
    `all_segments`. Segmentos sem o metadado disponível (retrocompatibilidade
    com dados anteriores a esta melhoria) são omitidos, nunca geram erro.
    """
    segments = {
        "by_lambda_tier": segment_by_lambda_tier(all_df),
        "by_effective_sample_size_range": segment_by_effective_sample_size_range(all_df),
        "by_model_confidence": segment_by_model_confidence(all_df),
    }
    return {name: result for name, result in segments.items() if not result.empty}
