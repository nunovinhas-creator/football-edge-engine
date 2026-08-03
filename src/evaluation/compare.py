"""
Comparação entre versões do Framework de Avaliação Quantitativa.

Compara dois ficheiros (ou DataFrames) de apostas já avaliadas — tal como
exportados por `BacktestReport.to_csv(...)` / `EvaluationReport.to_csv(...)`
em `bets.csv` — e responde às quatro perguntas pedidas:

    1. Qual modelo/versão ganhou mais lucro líquido?
    2. Qual teve maior ROI?
    3. Qual teve menor Brier Score?
    4. Qual teve melhor calibração (menor ECE)?

Não recalcula nenhuma métrica: reutiliza `evaluation.metrics.full_summary`,
que por sua vez reutiliza as fórmulas oficiais de
`src.backtest.historical`. Serve tipicamente para comparar duas execuções
do mesmo motor de previsão em datasets diferentes (ex. antes/depois de um
período, ou dois conjuntos de decisões históricas), não para comparar
fórmulas matemáticas distintas.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Union

import pandas as pd

from .formatting import dataframe_to_markdown
from .metrics import full_summary

BetsSource = Union[str, os.PathLike, pd.DataFrame]

# metric_key -> (label PT, chave dentro de metrics_a/metrics_b, "maior_melhor")
_COMPARISON_CRITERIA = {
    "profit": ("Lucro líquido (quem ganhou mais)", "net_profit", True),
    "roi": ("ROI (%)", "roi_pct", True),
    "brier": ("Brier Score (menor é melhor)", "brier_score", False),
    "calibration": ("Calibration Error / ECE (menor é melhor)", "calibration_error", False),
}


def load_bets(source: BetsSource) -> pd.DataFrame:
    """
    Carrega um ficheiro de apostas avaliadas (`bets.csv`) ou aceita
    diretamente um DataFrame já em memória. Não normaliza nem recalcula
    nenhuma coluna — assume o formato produzido por
    `src.backtest.historical.evaluator.evaluate_bets` /
    `BacktestReport.all_bets`.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _placed_subset(df: pd.DataFrame, placed_only: bool) -> pd.DataFrame:
    if placed_only and "placed" in df.columns:
        return df[df["placed"].astype(bool)].reset_index(drop=True)
    return df


def _winner(value_a: float, value_b: float, label_a: str, label_b: str, higher_is_better: bool) -> str:
    if value_a == value_b:
        return "EMPATE"
    if higher_is_better:
        return label_a if value_a > value_b else label_b
    return label_a if value_a < value_b else label_b


@dataclass
class ComparisonResult:
    """Resultado da comparação entre duas execuções de backtest."""

    label_a: str
    label_b: str
    metrics_a: Dict[str, Any]
    metrics_b: Dict[str, Any]

    def winner(self, criterion: str) -> str:
        """`criterion` é uma das chaves de `_COMPARISON_CRITERIA` ('profit', 'roi', 'brier', 'calibration')."""
        _, metric_key, higher_is_better = _COMPARISON_CRITERIA[criterion]
        return _winner(
            self.metrics_a[metric_key],
            self.metrics_b[metric_key],
            self.label_a,
            self.label_b,
            higher_is_better,
        )

    @property
    def winner_by_profit(self) -> str:
        return self.winner("profit")

    @property
    def winner_by_roi(self) -> str:
        return self.winner("roi")

    @property
    def winner_by_brier(self) -> str:
        return self.winner("brier")

    @property
    def winner_by_calibration(self) -> str:
        return self.winner("calibration")

    def summary(self) -> Dict[str, str]:
        """Dicionário com as quatro respostas pedidas pelo Framework de Avaliação."""
        return {
            "qual_modelo_ganhou_mais": self.winner_by_profit,
            "qual_teve_maior_roi": self.winner_by_roi,
            "qual_teve_menor_brier": self.winner_by_brier,
            "qual_teve_melhor_calibracao": self.winner_by_calibration,
        }

    def comparison_table(self) -> pd.DataFrame:
        """Tabela lado-a-lado com todas as métricas de `full_summary` para A e B."""
        keys = list(dict.fromkeys(list(self.metrics_a.keys()) + list(self.metrics_b.keys())))
        rows = [
            {"metric": key, self.label_a: self.metrics_a.get(key), self.label_b: self.metrics_b.get(key)}
            for key in keys
        ]
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        lines = [
            f"# Comparação de Backtests: {self.label_a} vs. {self.label_b}",
            "",
            "## Resultado",
            "",
        ]
        for criterion, (label, _, _) in _COMPARISON_CRITERIA.items():
            lines.append(f"- **{label}:** {self.winner(criterion)}")
        lines.append("")
        lines.append("## Métricas lado-a-lado")
        lines.append("")
        lines.append(dataframe_to_markdown(self.comparison_table()))
        lines.append("")
        return "\n".join(lines)


def compare_backtests(
    source_a: BetsSource,
    source_b: BetsSource,
    label_a: str = "A",
    label_b: str = "B",
    placed_only: bool = True,
    n_bins: int = 10,
) -> ComparisonResult:
    """
    Compara duas execuções de backtest a partir dos respetivos ficheiros
    `bets.csv` (ou DataFrames já carregados) e devolve um
    `ComparisonResult` com as métricas de cada uma e o "vencedor" em cada
    um dos quatro critérios pedidos (lucro, ROI, Brier, calibração).

    Métricas financeiras (ROI, lucro) são calculadas sobre as apostas
    colocadas (`placed_only=True`, por omissão); métricas estatísticas
    (Brier, ECE) são sempre calculadas sobre TODAS as apostas avaliadas de
    cada ficheiro, para medir a qualidade do modelo independentemente da
    estratégia de aposta.
    """
    df_a = load_bets(source_a)
    df_b = load_bets(source_b)

    metrics_a = full_summary(_placed_subset(df_a, placed_only), all_df=df_a, n_bins=n_bins)
    metrics_b = full_summary(_placed_subset(df_b, placed_only), all_df=df_b, n_bins=n_bins)

    return ComparisonResult(label_a=label_a, label_b=label_b, metrics_a=metrics_a, metrics_b=metrics_b)
