"""
Geração de relatórios do Backtesting Framework: tabela resumo, CSV, Excel
(opcional) e gráficos (evolução da banca, histogramas, curva de
calibração, distribuição de Edge/EV).

Este módulo é puramente de apresentação — não recalcula nenhuma métrica,
apenas formata e exporta o que já foi calculado por `metrics.py`,
`statistics.py`, `segments.py` e `thresholds.py`.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from . import statistics as stats_module
from .metrics import equity_curve, summary_metrics


@dataclass
class BacktestReport:
    """
    Resultado completo de uma execução do Backtesting Framework.

    all_bets:      todas as apostas avaliadas (independentemente de terem
                   sido "colocadas" pela decisão histórica do motor).
    placed_bets:   subconjunto de `all_bets` efetivamente apostado
                   (`engine_decision` continha "BET").
    global_metrics:      métricas financeiras (ver `metrics.summary_metrics`)
                         calculadas sobre `placed_bets`.
    statistical_metrics: Brier Score / Log Loss / ECE (ver
                         `statistics.statistical_summary`), calculadas
                         sobre `all_bets` (qualidade do modelo, não da
                         estratégia de aposta).
    calibration_curve:   curva de calibração (ver `statistics.calibration_curve`).
    segments:            {nome: DataFrame de métricas} por segmento.
    edge_thresholds / ev_thresholds: tabelas de threshold analysis.
    """

    all_bets: pd.DataFrame
    placed_bets: pd.DataFrame
    global_metrics: Dict[str, Any]
    statistical_metrics: Dict[str, Any]
    calibration_curve: pd.DataFrame
    segments: Dict[str, pd.DataFrame] = field(default_factory=dict)
    edge_thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)
    ev_thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # Tabela resumo
    # ------------------------------------------------------------------

    def summary_table(self) -> pd.DataFrame:
        """Combina métricas globais e estatísticas numa tabela de uma linha."""
        combined = {**self.global_metrics, **self.statistical_metrics}
        return pd.DataFrame([combined])

    def print_summary(self) -> None:
        summary = {**self.global_metrics, **self.statistical_metrics}
        width = max(len(k) for k in summary) + 2
        print("=" * (width + 20))
        print("BACKTEST — RESUMO DE DESEMPENHO")
        print("=" * (width + 20))
        for key, value in summary.items():
            print(f"{key:<{width}}: {value}")
        print("=" * (width + 20))

    # ------------------------------------------------------------------
    # Exportação
    # ------------------------------------------------------------------

    def to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Exporta o relatório completo como uma coleção de ficheiros CSV em
        `output_dir`. Devolve um dicionário {nome: caminho} dos ficheiros
        escritos.
        """
        os.makedirs(output_dir, exist_ok=True)
        written = {}

        paths = {
            "bets": os.path.join(output_dir, "bets.csv"),
            "summary": os.path.join(output_dir, "summary.csv"),
            "calibration_curve": os.path.join(output_dir, "calibration_curve.csv"),
            "edge_thresholds": os.path.join(output_dir, "edge_thresholds.csv"),
            "ev_thresholds": os.path.join(output_dir, "ev_thresholds.csv"),
        }

        self.all_bets.to_csv(paths["bets"], index=False)
        written["bets"] = paths["bets"]

        self.summary_table().to_csv(paths["summary"], index=False)
        written["summary"] = paths["summary"]

        self.calibration_curve.to_csv(paths["calibration_curve"], index=False)
        written["calibration_curve"] = paths["calibration_curve"]

        self.edge_thresholds.to_csv(paths["edge_thresholds"], index=False)
        written["edge_thresholds"] = paths["edge_thresholds"]

        self.ev_thresholds.to_csv(paths["ev_thresholds"], index=False)
        written["ev_thresholds"] = paths["ev_thresholds"]

        for name, segment_df in self.segments.items():
            segment_path = os.path.join(output_dir, f"segment_{name}.csv")
            segment_df.to_csv(segment_path, index=False)
            written[f"segment_{name}"] = segment_path

        return written

    def to_excel(self, path: str) -> str:
        """
        Exporta o relatório completo como um único ficheiro Excel
        (`.xlsx`), uma folha por secção. Requer `openpyxl` instalado
        (dependência opcional). Lança `ImportError` com uma mensagem clara
        caso não esteja disponível.
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Exportação para Excel requer o pacote opcional 'openpyxl' "
                "(pip install openpyxl)."
            ) from exc

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.summary_table().to_excel(writer, sheet_name="summary", index=False)
            self.all_bets.to_excel(writer, sheet_name="bets", index=False)
            self.calibration_curve.to_excel(writer, sheet_name="calibration", index=False)
            self.edge_thresholds.to_excel(writer, sheet_name="edge_thresholds", index=False)
            self.ev_thresholds.to_excel(writer, sheet_name="ev_thresholds", index=False)
            for name, segment_df in self.segments.items():
                # Nomes de folha do Excel são limitados a 31 caracteres.
                sheet_name = name[:31]
                segment_df.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------

    def generate_all_plots(self, output_dir: str) -> Dict[str, str]:
        """
        Gera todos os gráficos do relatório (evolução da banca,
        distribuição de probabilidades/Edge/EV, curva de calibração) como
        PNG em `output_dir`. Requer `matplotlib` (dependência já presente
        em requirements.txt).
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        written = {}

        # Evolução da banca (equity curve), apenas para as apostas colocadas.
        curve = equity_curve(self.placed_bets)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if not curve.empty:
            ax.plot(range(1, len(curve) + 1), curve.values, color="#1f6feb")
        ax.set_title("Evolução da Banca (Lucro Acumulado)")
        ax.set_xlabel("Número da Aposta")
        ax.set_ylabel("Lucro Acumulado")
        ax.axhline(0, color="gray", linewidth=0.8)
        fig.tight_layout()
        path = os.path.join(output_dir, "equity_curve.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written["equity_curve"] = path

        # Distribuições
        for name, column, title in [
            ("probability_distribution", "probability", "Distribuição das Probabilidades"),
            ("edge_distribution", "edge", "Distribuição do Edge"),
            ("ev_distribution", "ev", "Distribuição do EV"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if not self.all_bets.empty:
                ax.hist(self.all_bets[column].dropna(), bins=15, color="#1f6feb", edgecolor="white")
            ax.set_title(title)
            fig.tight_layout()
            path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written[name] = path

        # Curva de calibração (reliability diagram)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibração perfeita")
        if not self.calibration_curve.empty:
            ax.plot(
                self.calibration_curve["predicted_mean"],
                self.calibration_curve["actual_frequency"],
                marker="o",
                color="#da3633",
                label="Modelo",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Probabilidade Prevista")
        ax.set_ylabel("Frequência Real de Acerto")
        ax.set_title("Curva de Calibração")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, "calibration_curve.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written["calibration_curve"] = path

        return written
