"""
Gráficos adicionais do Framework de Avaliação Quantitativa.

`src.backtest.historical.report.BacktestReport.generate_all_plots` já
produz: evolução da banca (lucro acumulado absoluto), distribuição de
probabilidades/Edge/EV e curva de calibração. Este módulo acrescenta
apenas os gráficos pedidos que ainda não existiam — ROI acumulado (%),
evolução da banca em unidades monetárias a partir de uma banca inicial,
distribuição de odds, lucro por competição, yield por mercado, e o
"reliability diagram" (matematicamente idêntico à curva de calibração,
exportado também com este nome por clareza de nomenclatura).

Puramente de apresentação: nenhuma fórmula é recalculada — reutiliza
`evaluation.metrics` / `evaluation.segments`, que por sua vez reutilizam
`src.backtest.historical`. Requer `matplotlib` (já em requirements.txt).
"""

import os
from typing import Dict

import pandas as pd

from .metrics import calibration_curve as _calibration_curve
from .segments import segment_by_column

__all__ = [
    "cumulative_roi_series",
    "bankroll_series",
    "plot_cumulative_roi",
    "plot_bankroll",
    "plot_odds_distribution",
    "plot_profit_by_competition",
    "plot_yield_by_market",
    "plot_reliability_diagram",
    "generate_extra_plots",
]


def cumulative_roi_series(df: pd.DataFrame) -> pd.Series:
    """
    ROI acumulado (%) após cada aposta: lucro acumulado / stake acumulado
    * 100, na ordem em que as apostas aparecem no DataFrame (cronológica,
    ver `evaluator.evaluate_bets`).
    """
    if df.empty:
        return pd.Series(dtype=float)
    cum_profit = df["profit"].cumsum()
    cum_stake = df["stake"].cumsum().replace(0, pd.NA)
    return (100.0 * cum_profit / cum_stake).astype(float)


def bankroll_series(df: pd.DataFrame, starting_bankroll: float = 1000.0) -> pd.Series:
    """Banca acumulada em unidades monetárias, partindo de `starting_bankroll`."""
    if df.empty:
        return pd.Series(dtype=float)
    return starting_bankroll + df["profit"].cumsum()


def _new_figure(figsize=(9, 4.5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(figsize=figsize)


def plot_cumulative_roi(df: pd.DataFrame, path: str) -> str:
    """Evolução do ROI acumulado (%) ao longo das apostas."""
    plt, (fig, ax) = _new_figure()
    series = cumulative_roi_series(df)
    if not series.empty:
        ax.plot(range(1, len(series) + 1), series.values, color="#1f6feb")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("ROI Acumulado (%)")
    ax.set_xlabel("Número da Aposta")
    ax.set_ylabel("ROI Acumulado (%)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_bankroll(df: pd.DataFrame, path: str, starting_bankroll: float = 1000.0) -> str:
    """Evolução da banca (unidades monetárias) partindo de `starting_bankroll`."""
    plt, (fig, ax) = _new_figure()
    series = bankroll_series(df, starting_bankroll=starting_bankroll)
    if not series.empty:
        ax.plot(range(1, len(series) + 1), series.values, color="#238636")
    ax.axhline(starting_bankroll, color="gray", linewidth=0.8, linestyle="--", label="Banca inicial")
    ax.set_title("Evolução da Banca")
    ax.set_xlabel("Número da Aposta")
    ax.set_ylabel("Banca")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_odds_distribution(df: pd.DataFrame, path: str) -> str:
    """Histograma da distribuição de odds das apostas avaliadas."""
    plt, (fig, ax) = _new_figure(figsize=(7, 4.5))
    if not df.empty:
        ax.hist(df["odd"].dropna(), bins=15, color="#8957e5", edgecolor="white")
    ax.set_title("Distribuição de Odds")
    ax.set_xlabel("Odd")
    ax.set_ylabel("Nº de Apostas")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_profit_by_competition(df: pd.DataFrame, path: str) -> str:
    """Lucro líquido total, por competição (apostas colocadas)."""
    segment = segment_by_column(df, "competition")
    plt, (fig, ax) = _new_figure(figsize=(9, 4.5))
    if not segment.empty:
        ordered = segment.sort_values("net_profit")
        colors = ["#da3633" if v < 0 else "#238636" for v in ordered["net_profit"]]
        ax.barh(ordered["competition"].astype(str), ordered["net_profit"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Lucro por Competição")
    ax.set_xlabel("Lucro Líquido")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_yield_by_market(df: pd.DataFrame, path: str) -> str:
    """Yield (%), por mercado (apostas colocadas)."""
    segment = segment_by_column(df, "market")
    plt, (fig, ax) = _new_figure(figsize=(9, 4.5))
    if not segment.empty:
        ordered = segment.sort_values("yield_pct")
        colors = ["#da3633" if v < 0 else "#1f6feb" for v in ordered["yield_pct"]]
        ax.barh(ordered["market"].astype(str), ordered["yield_pct"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Yield por Mercado")
    ax.set_xlabel("Yield (%)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_reliability_diagram(df: pd.DataFrame, path: str, n_bins: int = 10) -> str:
    """
    Reliability diagram: probabilidade prevista vs. frequência real de
    acerto, por contentor. Matematicamente idêntico à "Curva de
    Calibração" já gerada por `BacktestReport.generate_all_plots`
    (`statistics.calibration_curve`) — exportado também sob este nome
    porque ambos os termos são pedidos explicitamente pelo Framework de
    Avaliação.
    """
    curve = _calibration_curve(df, n_bins=n_bins)
    plt, (fig, ax) = _new_figure(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibração perfeita")
    if not curve.empty:
        ax.plot(curve["predicted_mean"], curve["actual_frequency"], marker="o", color="#da3633", label="Modelo")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidade Prevista")
    ax.set_ylabel("Frequência Real de Acerto")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_extra_plots(
    placed_bets: pd.DataFrame,
    all_bets: pd.DataFrame,
    output_dir: str,
    starting_bankroll: float = 1000.0,
) -> Dict[str, str]:
    """
    Gera todos os gráficos adicionais deste módulo em `output_dir` e
    devolve {nome: caminho}. `placed_bets` alimenta os gráficos
    financeiros (ROI acumulado, banca, lucro por competição, yield por
    mercado); `all_bets` alimenta os gráficos de qualidade do modelo
    (distribuição de odds, reliability diagram), tal como no resto do
    Framework de Avaliação.
    """
    os.makedirs(output_dir, exist_ok=True)
    return {
        "cumulative_roi": plot_cumulative_roi(placed_bets, os.path.join(output_dir, "cumulative_roi.png")),
        "bankroll": plot_bankroll(
            placed_bets, os.path.join(output_dir, "bankroll.png"), starting_bankroll=starting_bankroll
        ),
        "odds_distribution": plot_odds_distribution(all_bets, os.path.join(output_dir, "odds_distribution.png")),
        "profit_by_competition": plot_profit_by_competition(
            placed_bets, os.path.join(output_dir, "profit_by_competition.png")
        ),
        "yield_by_market": plot_yield_by_market(placed_bets, os.path.join(output_dir, "yield_by_market.png")),
        "reliability_diagram": plot_reliability_diagram(
            all_bets, os.path.join(output_dir, "reliability_diagram.png")
        ),
    }
