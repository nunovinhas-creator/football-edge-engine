"""
Relatório centralizado do Framework de Avaliação Quantitativa.

`EvaluationReport` envolve um `BacktestReport`
(`src.backtest.historical.report.BacktestReport`) sem alterar nenhuma das
suas métricas, e acrescenta apenas o que faltava para cumprir os
requisitos do Framework de Avaliação:

    - segmentos adicionais: por mês e por faixa de confiança
      (`evaluation.segments`);
    - exportação em Markdown (as exportações CSV/Excel/HTML já existiam
      em `BacktestReport` e são reutilizadas tal como estão);
    - gráficos adicionais: ROI acumulado, banca, distribuição de odds,
      lucro por competição, yield por mercado, reliability diagram
      (`evaluation.plots`), a somar aos já produzidos por
      `BacktestReport.generate_all_plots`.

Não recalcula nem altera nenhuma fórmula do motor de previsão nem do
Backtesting Framework — é puramente uma camada de orquestração e
apresentação.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from src.backtest.historical.engine import BacktestEngine, BetsInput
from src.backtest.historical.report import BacktestReport
from src.backtest.historical.staking import FlatStake, StakingStrategy

from . import plots as plots_module
from .formatting import dataframe_to_markdown
from .segments import all_segments as _all_segments

__all__ = ["EvaluationReport", "evaluate"]


@dataclass
class EvaluationReport:
    """
    Relatório completo do Framework de Avaliação Quantitativa: envolve um
    `BacktestReport` (métricas financeiras/estatísticas, segmentos
    "clássicos" e threshold analysis já existentes) e acrescenta os
    segmentos por mês/confiança, a exportação Markdown e os gráficos
    adicionais pedidos pelos requisitos.
    """

    backtest_report: BacktestReport
    extra_segments: Dict[str, pd.DataFrame] = field(default_factory=dict)
    starting_bankroll: float = 1000.0

    # ------------------------------------------------------------------
    # Acesso direto aos dados subjacentes (delegação, sem recalcular nada)
    # ------------------------------------------------------------------

    @property
    def all_bets(self) -> pd.DataFrame:
        return self.backtest_report.all_bets

    @property
    def placed_bets(self) -> pd.DataFrame:
        return self.backtest_report.placed_bets

    @property
    def global_metrics(self) -> Dict[str, Any]:
        return self.backtest_report.global_metrics

    @property
    def statistical_metrics(self) -> Dict[str, Any]:
        return self.backtest_report.statistical_metrics

    def all_segment_tables(self) -> Dict[str, pd.DataFrame]:
        """Todos os segmentos (competição/mercado/odd/edge/ev/favorito/casa-fora/mês/confiança)."""
        return {**self.backtest_report.segments, **self.extra_segments}

    def summary_table(self) -> pd.DataFrame:
        return self.backtest_report.summary_table()

    def print_summary(self) -> None:
        self.backtest_report.print_summary()

    @classmethod
    def from_backtest_report(cls, report: BacktestReport, starting_bankroll: float = 1000.0) -> "EvaluationReport":
        """Constrói um `EvaluationReport` a partir de um `BacktestReport` já calculado."""
        extra_segments = {}
        segments = _all_segments(report.placed_bets)
        for name in ("by_month", "by_confidence_range"):
            if name in segments:
                extra_segments[name] = segments[name]
        return cls(backtest_report=report, extra_segments=extra_segments, starting_bankroll=starting_bankroll)

    # ------------------------------------------------------------------
    # Exportação
    # ------------------------------------------------------------------

    def to_csv(self, output_dir: str) -> Dict[str, str]:
        """CSV: reutiliza `BacktestReport.to_csv` e acrescenta os segmentos extra (mês/confiança)."""
        written = self.backtest_report.to_csv(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for name, segment_df in self.extra_segments.items():
            path = os.path.join(output_dir, f"segment_{name}.csv")
            segment_df.to_csv(path, index=False)
            written[f"segment_{name}"] = path
        return written

    def to_excel(self, path: str) -> str:
        """
        Excel: um único workbook com tudo o que `BacktestReport.to_excel`
        já produz, mais uma folha por segmento extra (mês/confiança).
        Requer `openpyxl` (dependência opcional já usada por
        `BacktestReport`).
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Exportação para Excel requer o pacote opcional 'openpyxl' (pip install openpyxl)."
            ) from exc

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.summary_table().to_excel(writer, sheet_name="summary", index=False)
            self.all_bets.to_excel(writer, sheet_name="bets", index=False)
            self.backtest_report.calibration_curve.to_excel(writer, sheet_name="calibration", index=False)
            self.backtest_report.edge_thresholds.to_excel(writer, sheet_name="edge_thresholds", index=False)
            self.backtest_report.ev_thresholds.to_excel(writer, sheet_name="ev_thresholds", index=False)
            for name, segment_df in self.all_segment_tables().items():
                sheet_name = name[:31]  # limite do Excel
                segment_df.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    def to_html(self, path: str, plots_dir: Optional[str] = None, title: str = "Relatório de Avaliação") -> str:
        """
        HTML autocontido com o resumo, threshold analysis, TODOS os
        segmentos (clássicos + mês/confiança) e, se `plots_dir` for
        indicado, todos os gráficos (`BacktestReport` + `evaluation.plots`)
        embutidos como imagens base64.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def _df_html(df: Optional[pd.DataFrame]) -> str:
            if df is None or df.empty:
                return "<p><em>Sem dados.</em></p>"
            return df.to_html(index=False, border=0, classes="table", na_rep="-")

        sections = [f"<h1>{title}</h1>"]

        sections.append("<h2>Resumo Global</h2>")
        sections.append(_df_html(self.summary_table()))

        sections.append("<h2>Threshold Analysis — Edge</h2>")
        sections.append(_df_html(self.backtest_report.edge_thresholds))

        sections.append("<h2>Threshold Analysis — EV</h2>")
        sections.append(_df_html(self.backtest_report.ev_thresholds))

        sections.append("<h2>Curva de Calibração</h2>")
        sections.append(_df_html(self.backtest_report.calibration_curve))

        sections.append("<h2>Segmentos</h2>")
        for name, segment_df in self.all_segment_tables().items():
            sections.append(f"<h3>{name}</h3>")
            sections.append(_df_html(segment_df))

        if plots_dir and os.path.isdir(plots_dir):
            import base64

            sections.append("<h2>Gráficos</h2>")
            for filename in sorted(os.listdir(plots_dir)):
                if not filename.lower().endswith(".png"):
                    continue
                with open(os.path.join(plots_dir, filename), "rb") as fh:
                    encoded = base64.b64encode(fh.read()).decode("ascii")
                sections.append(f"<h3>{filename}</h3>")
                sections.append(f'<img alt="{filename}" src="data:image/png;base64,{encoded}">')

        sections.append("<h2>Apostas (todas)</h2>")
        sections.append(_df_html(self.all_bets))

        body = "\n".join(sections)
        html = f"""<!doctype html>
<html lang="pt">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
h1, h2, h3 {{ color: #0d1117; }}
table.table {{ border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.85rem; }}
table.table th, table.table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
table.table th {{ background: #f0f2f5; }}
img {{ max-width: 100%; margin-bottom: 1.5rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path

    def to_markdown(self, path: str, title: str = "Relatório de Avaliação") -> str:
        """
        Relatório Markdown autocontido: resumo global + estatístico,
        threshold analysis e todos os segmentos (clássicos + mês/confiança).
        Formato pedido pelo Framework de Avaliação que ainda não existia
        em `BacktestReport`.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        lines = [f"# {title}", ""]

        lines.append("## Resumo Global")
        lines.append("")
        lines.append(dataframe_to_markdown(self.summary_table()))
        lines.append("")

        lines.append("## Threshold Analysis — Edge")
        lines.append("")
        lines.append(dataframe_to_markdown(self.backtest_report.edge_thresholds))
        lines.append("")

        lines.append("## Threshold Analysis — EV")
        lines.append("")
        lines.append(dataframe_to_markdown(self.backtest_report.ev_thresholds))
        lines.append("")

        lines.append("## Curva de Calibração")
        lines.append("")
        lines.append(dataframe_to_markdown(self.backtest_report.calibration_curve))
        lines.append("")

        lines.append("## Segmentos")
        lines.append("")
        for name, segment_df in self.all_segment_tables().items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(dataframe_to_markdown(segment_df))
            lines.append("")

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------

    def generate_all_plots(self, output_dir: str) -> Dict[str, str]:
        """
        Todos os gráficos: os já existentes em `BacktestReport`
        (evolução da banca absoluta, distribuições de probabilidade/edge/
        ev, curva de calibração) mais os novos deste módulo (ROI
        acumulado, banca a partir de `starting_bankroll`, distribuição de
        odds, lucro por competição, yield por mercado, reliability
        diagram).
        """
        written = self.backtest_report.generate_all_plots(output_dir)
        written.update(
            plots_module.generate_extra_plots(
                self.placed_bets,
                self.all_bets,
                output_dir,
                starting_bankroll=self.starting_bankroll,
            )
        )
        return written


def evaluate(
    bets: BetsInput,
    staking: Optional[StakingStrategy] = None,
    starting_bankroll: float = 1000.0,
    **engine_kwargs: Any,
) -> EvaluationReport:
    """
    Ponto de entrada único do Framework de Avaliação Quantitativa: corre o
    `BacktestEngine` já existente (sem alterar nenhuma fórmula) sobre
    `bets` e devolve um `EvaluationReport` completo (métricas, segmentos
    incluindo mês/confiança, e exportação CSV/Excel/HTML/Markdown +
    gráficos).
    """
    engine = BacktestEngine(staking=staking or FlatStake(unit=1.0), **engine_kwargs)
    backtest_report = engine.run(bets)
    return EvaluationReport.from_backtest_report(backtest_report, starting_bankroll=starting_bankroll)
