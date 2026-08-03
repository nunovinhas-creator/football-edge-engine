"""
Métricas globais de desempenho do Backtesting Framework.

Todas as funções recebem um DataFrame de apostas já avaliadas (ver
`evaluator.evaluate_bets`) e já filtradas para o subconjunto de interesse
(ex. apenas apostas "colocadas", ou um segmento, ou um threshold de Edge).
Isto torna estas funções reutilizáveis por `segments.py` e `thresholds.py`
sem duplicar lógica.

Convenções:
    - `edge`, `ev`, `kelly` são frações (0.0-1.0), tal como devolvidos por
      `src.engine.edge` / `src.engine.kelly`. As médias devolvidas por este
      módulo em "_pct" já vêm multiplicadas por 100 para apresentação.
    - ROI é ponderado pelo stake total (capital investido).
    - Yield é a média das rentabilidades individuais por aposta
      (profit_i / stake_i), não ponderada pelo tamanho do stake — por isso
      diverge de ROI sempre que o stake varia entre apostas (ex. Kelly).
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["odd", "probability", "edge", "ev", "kelly", "stake", "won", "profit"]


def _empty_frame_metrics() -> Dict[str, Any]:
    return {
        "n_bets": 0,
        "wins": 0,
        "losses": 0,
        "hit_rate_pct": 0.0,
        "total_staked": 0.0,
        "net_profit": 0.0,
        "roi_pct": 0.0,
        "yield_pct": 0.0,
        "avg_odd": 0.0,
        "avg_edge_pct": 0.0,
        "avg_ev_pct": 0.0,
        "avg_kelly_pct": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "expectancy_per_bet": 0.0,
    }


def hit_rate(df: pd.DataFrame) -> float:
    """Taxa de acerto (%) do subconjunto de apostas."""
    if df.empty:
        return 0.0
    return round(100.0 * df["won"].sum() / len(df), 2)


def net_profit(df: pd.DataFrame) -> float:
    """Lucro líquido total (soma de `profit` por aposta)."""
    if df.empty:
        return 0.0
    return round(float(df["profit"].sum()), 4)


def total_staked(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(float(df["stake"].sum()), 4)


def roi(df: pd.DataFrame) -> float:
    """
    ROI (%) = lucro líquido total / total apostado * 100.
    Ponderado pelo stake — reflete o retorno sobre o capital investido.
    """
    staked = total_staked(df)
    if staked <= 0:
        return 0.0
    return round(100.0 * net_profit(df) / staked, 2)


def yield_pct(df: pd.DataFrame) -> float:
    """
    Yield (%) = média das rentabilidades individuais (profit_i / stake_i).
    Não ponderada pelo stake — distingue-se do ROI quando o stake varia
    (ex. staking baseado em Kelly, onde apostas com maior edge recebem
    maior stake).
    """
    if df.empty:
        return 0.0
    valid = df[df["stake"] > 0]
    if valid.empty:
        return 0.0
    per_bet_return = valid["profit"] / valid["stake"]
    return round(100.0 * float(per_bet_return.mean()), 2)


def equity_curve(df: pd.DataFrame) -> pd.Series:
    """Banca acumulada (lucro líquido cumulativo), ordenada pela ordem do DataFrame."""
    if df.empty:
        return pd.Series(dtype=float)
    return df["profit"].cumsum()


def max_drawdown(df: pd.DataFrame) -> Dict[str, float]:
    """
    Drawdown máximo, em valor absoluto e em percentagem do pico de banca
    acumulada. Assume que o DataFrame já está ordenado cronologicamente
    (ver `evaluator.evaluate_bets`, que ordena por `date`).
    """
    if df.empty:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}

    curve = equity_curve(df)
    running_peak = curve.cummax()
    drawdown_series = curve - running_peak
    max_dd = float(drawdown_series.min())

    peak_at_max_dd = float(running_peak.loc[drawdown_series.idxmin()])
    if peak_at_max_dd > 0:
        max_dd_pct = 100.0 * max_dd / peak_at_max_dd
    else:
        max_dd_pct = 0.0

    return {
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
    }


def profit_factor(df: pd.DataFrame) -> float:
    """
    Profit Factor = soma dos lucros positivos / |soma dos lucros negativos|.
    Devolve `float('inf')` se não existirem perdas (todas as apostas
    ganhadoras).
    """
    if df.empty:
        return 0.0
    gains = df.loc[df["profit"] > 0, "profit"].sum()
    losses = df.loc[df["profit"] < 0, "profit"].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return round(float(gains / abs(losses)), 4)


def expectancy_per_bet(df: pd.DataFrame) -> float:
    """Expectativa por aposta: lucro líquido médio por aposta (unidades monetárias)."""
    if df.empty:
        return 0.0
    return round(float(df["profit"].mean()), 4)


def summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Agrega todas as métricas globais num único dicionário."""
    if df.empty:
        return _empty_frame_metrics()

    dd = max_drawdown(df)
    return {
        "n_bets": int(len(df)),
        "wins": int(df["won"].sum()),
        "losses": int((~df["won"]).sum()),
        "hit_rate_pct": hit_rate(df),
        "total_staked": total_staked(df),
        "net_profit": net_profit(df),
        "roi_pct": roi(df),
        "yield_pct": yield_pct(df),
        "avg_odd": round(float(df["odd"].mean()), 4),
        "avg_edge_pct": round(float(df["edge"].mean()) * 100, 2),
        "avg_ev_pct": round(float(df["ev"].mean()) * 100, 2),
        "avg_kelly_pct": round(float(df["kelly"].mean()) * 100, 2),
        "max_drawdown": dd["max_drawdown"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "profit_factor": profit_factor(df),
        "expectancy_per_bet": expectancy_per_bet(df),
    }
