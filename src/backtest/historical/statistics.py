"""
Métricas estatísticas de qualidade probabilística do modelo.

Ao contrário de `metrics.py` (desempenho financeiro), este módulo avalia
a CALIBRAÇÃO do modelo: o quão próximas estão as probabilidades previstas
da frequência real de acerto — independentemente de a aposta ter sido
colocada ou não. Por omissão deve ser usado sobre TODAS as apostas
avaliadas (`evaluate_bets`), não apenas as apostas colocadas, pois mede a
qualidade do modelo, não a rentabilidade da estratégia.

Nenhuma destas fórmulas depende de scipy — apenas numpy/pandas, para
manter as dependências do projeto mínimas.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

_EPSILON = 1e-12


def brier_score(df: pd.DataFrame) -> float:
    """
    Brier Score = média((probabilidade_prevista - resultado_real)^2).
    Quanto mais baixo, melhor a calibração (0 = perfeito, 1 = pior possível).
    """
    if df.empty:
        return 0.0
    y = df["won"].astype(float)
    p = df["probability"].astype(float)
    return round(float(np.mean((p - y) ** 2)), 6)


def log_loss(df: pd.DataFrame) -> float:
    """
    Log Loss (entropia cruzada binária) = -média(y*log(p) + (1-y)*log(1-p)).
    As probabilidades são "clipadas" para [epsilon, 1-epsilon] para evitar
    log(0).
    """
    if df.empty:
        return 0.0
    y = df["won"].astype(float).to_numpy()
    p = np.clip(df["probability"].astype(float).to_numpy(), _EPSILON, 1 - _EPSILON)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return round(float(np.mean(losses)), 6)


def calibration_curve(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Curva de calibração: divide as probabilidades previstas em `n_bins`
    contentores de igual largura em [0, 1] e, para cada um, calcula a
    probabilidade média prevista vs. a frequência real de acerto.

    Devolve um DataFrame com colunas:
        bin_low, bin_high, predicted_mean, actual_frequency, count
    Contentores sem observações são omitidos.
    """
    columns = ["bin_low", "bin_high", "predicted_mean", "actual_frequency", "count"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    p = df["probability"].astype(float).to_numpy()
    y = df["won"].astype(float).to_numpy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(p, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_low": round(float(bin_edges[i]), 4),
                "bin_high": round(float(bin_edges[i + 1]), 4),
                "predicted_mean": round(float(p[mask].mean()), 4),
                "actual_frequency": round(float(y[mask].mean()), 4),
                "count": count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def calibration_error(df: pd.DataFrame, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE): média ponderada (pelo nº de
    observações por contentor) da diferença absoluta entre a probabilidade
    prevista média e a frequência real de acerto.
    """
    curve = calibration_curve(df, n_bins=n_bins)
    if curve.empty:
        return 0.0
    total = curve["count"].sum()
    weighted_gap = (curve["predicted_mean"] - curve["actual_frequency"]).abs() * curve["count"]
    return round(float(weighted_gap.sum() / total), 6)


def _distribution(values: pd.Series, n_bins: int) -> Dict[str, Any]:
    values = values.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return {"bin_edges": [], "counts": [], "values": []}
    counts, bin_edges = np.histogram(values, bins=n_bins)
    return {
        "bin_edges": [round(float(edge), 6) for edge in bin_edges],
        "counts": [int(c) for c in counts],
        "values": values.tolist(),
    }


def probability_distribution(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    """Distribuição das probabilidades previstas pelo modelo (histograma)."""
    if df.empty:
        return {"bin_edges": [], "counts": [], "values": []}
    return _distribution(df["probability"], n_bins)


def edge_distribution(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    """Distribuição do Edge (fração) entre as apostas avaliadas."""
    if df.empty:
        return {"bin_edges": [], "counts": [], "values": []}
    return _distribution(df["edge"], n_bins)


def ev_distribution(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    """Distribuição do EV (fração) entre as apostas avaliadas."""
    if df.empty:
        return {"bin_edges": [], "counts": [], "values": []}
    return _distribution(df["ev"], n_bins)


def statistical_summary(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    """Agrega Brier Score, Log Loss e ECE num único dicionário."""
    return {
        "brier_score": brier_score(df),
        "log_loss": log_loss(df),
        "calibration_error": calibration_error(df, n_bins=n_bins),
    }
