"""
Framework de Avaliação Quantitativa do Football Edge Engine.

**Âmbito:** módulo puramente de MEDIÇÃO. Não altera, recalcula nem
substitui nenhum modelo matemático de previsão — Poisson, Dixon-Coles,
Monte Carlo, estimativa de λ, Kelly, Edge/EV, Goal Engine, Machine
Learning ou o Decision Engine permanecem exatamente como estão. Centraliza
e reutiliza as métricas já implementadas em `src.backtest.historical`
(ROI, Yield, Profit, Hit Rate, Brier Score, Log Loss, ECE, segmentos,
threshold analysis) e acrescenta apenas o que faltava para responder à
pergunta "o sistema está a melhorar?": segmentação por mês/confiança,
comparação objetiva entre duas execuções de backtest, exportação Markdown
e um punhado de gráficos adicionais.

Localização
-----------
    src/evaluation/
        metrics.py      # ROI, Yield, Profit, Hit Rate, Brier, LogLoss, ECE, avg odd/EV, stake total, n_bets
        segments.py      # todas as análises por segmento (+ mês e faixa de confiança, novos)
        compare.py         # comparação objetiva entre dois ficheiros de backtest
        plots.py             # gráficos adicionais (ROI acumulado, banca, odds, lucro/comp., yield/mercado, reliability)
        report.py              # EvaluationReport (CSV/Excel/HTML/Markdown) + evaluate()
        formatting.py             # utilitário de tabela Markdown partilhado

Uso típico
----------
    from src.evaluation import evaluate
    from src.backtest.historical.dataset import load_historical_dataset

    dados = load_historical_dataset("examples/backtest/sample_real_games.csv")
    report = evaluate(dados)

    report.print_summary()
    report.to_csv("output/evaluation")
    report.to_excel("output/evaluation/report.xlsx")
    report.generate_all_plots("output/evaluation/plots")
    report.to_html("output/evaluation/report.html", plots_dir="output/evaluation/plots")
    report.to_markdown("output/evaluation/report.md")

Comparar duas versões (dois ficheiros `bets.csv` já exportados):

    from src.evaluation import compare_backtests

    resultado = compare_backtests("v1/bets.csv", "v2/bets.csv", label_a="v1", label_b="v2")
    print(resultado.summary())
"""

from .compare import ComparisonResult, compare_backtests, load_bets
from .metrics import full_summary
from .report import EvaluationReport, evaluate
from .segments import all_segments

__all__ = [
    "evaluate",
    "EvaluationReport",
    "full_summary",
    "all_segments",
    "compare_backtests",
    "ComparisonResult",
    "load_bets",
]
