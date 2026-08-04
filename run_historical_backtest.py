#!/usr/bin/env python3
"""
CLI de ponta a ponta: dataset do Historical Dataset Builder (BSD API) ->
Backtesting Framework -> Framework de Avaliação Quantitativa.

Não altera nenhum algoritmo de previsão nem recalcula nenhuma fórmula:

    1. Lê um dataset já produzido por `build_historical_dataset.py`
       (`src.historical_dataset` — jogos terminados reais, com odds e
       resultado final, obtidos da BSD API).
    2. Converte-o para o formato aceite pelo Backtesting Framework já
       existente via `src.historical_dataset.backtest_bridge`, incluindo
       `model_prob`: a probabilidade 1X2 vem do MESMO Dixon-Coles já em
       produção para jogos futuros (`src.engine.lambda_estimator` +
       `src.engine.value.estimate_pregame_probabilities`,
       ver `docs/AUDIT_MATEMATICA.md` §15/§16), aplicado aqui a jogos
       terminados com `head_to_head` derivado apenas de confrontos
       diretos ANTERIORES dentro do próprio dataset (sem fuga de
       informação — `backtest_bridge.derive_h2h`).
    3. Corre `src.backtest.historical.BacktestEngine` (Edge/EV/Kelly já
       existentes) e `src.evaluation.report.evaluate` (Framework de
       Avaliação: ROI, Yield, Profit, Brier Score, Log Loss, ECE,
       Drawdown, segmentos, exportação) sobre o resultado.

Uso:

    # 1. Construir o dataset histórico real (BSD API)
    python build_historical_dataset.py --output-dir data/historical --competition-id 8

    # 2. Backtest + avaliação de ponta a ponta sobre esse dataset
    python run_historical_backtest.py --input data/historical/historical.csv
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.backtest.historical.staking import FlatStake, KellyStake
from src.backtest.historical.dataset import load_historical_dataset
from src.evaluation.report import evaluate
from src.historical_dataset.backtest_bridge import (
    model_probabilities_from_dixon_coles,
    to_backtest_frame,
)

# Mercados 1X2 suportados por `src.engine.value.estimate_pregame_probabilities`
# (Dixon-Coles) sem qualquer agregação nova sobre a matriz de resultados —
# Over/Under e BTTS ficariam fora do âmbito desta ligação (exigiriam somar a
# matriz de outra forma, o que não está pedido).
DEFAULT_MARKETS = ["HOME", "DRAW", "AWAY"]
MARKET_TO_PROB_KEY = {"HOME": "home", "DRAW": "draw", "AWAY": "away"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Corre o Backtesting Framework + Framework de Avaliação sobre um dataset "
            "real produzido pelo Historical Dataset Builder (BSD API)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="CSV produzido por build_historical_dataset.py (dataset normalizado, um jogo por linha).",
    )
    parser.add_argument(
        "--markets", type=str, default=",".join(DEFAULT_MARKETS),
        help=f"Mercados 1X2 a incluir no backtest, separados por vírgula (por omissão: {','.join(DEFAULT_MARKETS)}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/historical_backtest",
        help="Diretório onde gravar o relatório (CSV, Excel, HTML, Markdown e gráficos).",
    )
    parser.add_argument(
        "--staking", choices=["flat", "kelly"], default="flat",
        help="Estratégia de stake (dimensiona a aposta; não altera Edge/EV/Kelly).",
    )
    parser.add_argument("--unit", type=float, default=1.0, help="Stake fixo, quando --staking=flat.")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Fração de Kelly, quando --staking=kelly.")
    parser.add_argument("--kelly-cap", type=float, default=0.05, help="Stake máximo (fração da banca), quando --staking=kelly.")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Banca de referência (staking Kelly e gráfico de banca).")
    parser.add_argument("--min-edge", type=float, default=3.0, help="Edge mínimo (%%) usado pelo DecisionEngine para decidir 'BET'.")
    parser.add_argument("--max-kelly-fraction", type=float, default=0.25, help="Fração de Kelly máxima usada pelo DecisionEngine.")
    parser.add_argument("--no-excel", action="store_true", help="Não gera o ficheiro Excel.")
    parser.add_argument("--no-plots", action="store_true", help="Não gera os gráficos (nem os embute no HTML).")
    parser.add_argument("--no-html", action="store_true", help="Não gera o relatório HTML.")
    parser.add_argument("--no-markdown", action="store_true", help="Não gera o relatório Markdown.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    unsupported = [m for m in markets if m not in MARKET_TO_PROB_KEY]
    if unsupported:
        parser.error(f"Mercados não suportados por esta ligação (Dixon-Coles 1X2): {unsupported}")

    print(f"A carregar dataset do Historical Dataset Builder: {args.input}")
    records = pd.read_csv(args.input).to_dict(orient="records")
    print(f"Jogos obtidos do Historical Dataset Builder: {len(records)}")
    if not records:
        print("Dataset vazio — nada para avaliar.", file=sys.stderr)
        sys.exit(1)

    print("A calcular probabilidades do modelo (Dixon-Coles em produção, head-to-head derivado do próprio dataset)...")
    probabilities = model_probabilities_from_dixon_coles(records)

    frames = []
    for market in markets:
        prob_key = MARKET_TO_PROB_KEY[market]
        frame = to_backtest_frame(
            records, market=market,
            model_prob=lambda row, key=prob_key: probabilities.get(row["event_id"], {}).get(key),
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = combined.dropna(subset=["model_prob"]).reset_index(drop=True)
    n_games = combined[["home_team", "away_team", "date"]].drop_duplicates().shape[0] if not combined.empty else 0
    print(f"Jogos com odd e probabilidade válidas para pelo menos um mercado: {n_games}")
    print(f"Apostas simuladas (jogo x mercado): {len(combined)}")

    if combined.empty:
        print("Nenhuma aposta simulável (sem odds reais para os mercados/jogos pedidos).", file=sys.stderr)
        sys.exit(1)

    dataset = load_historical_dataset(
        combined,
        max_kelly_fraction=args.max_kelly_fraction,
        min_edge=args.min_edge,
    )

    if args.staking == "kelly":
        staking = KellyStake(fraction=args.kelly_fraction, cap=args.kelly_cap, bankroll=args.bankroll)
    else:
        staking = FlatStake(unit=args.unit)

    report = evaluate(dataset, staking=staking, starting_bankroll=args.bankroll)
    report.print_summary()

    gm = report.global_metrics
    sm = report.statistical_metrics
    print("\n=== Métricas pedidas (Melhoria #4) ===")
    print(f"Jogos analisados: {n_games}")
    print(f"Apostas simuladas: {len(dataset)}")
    print(f"Apostas colocadas (engine_decision=BET): {gm['n_bets']}")
    print(f"ROI: {gm['roi_pct']}%")
    print(f"Yield: {gm['yield_pct']}%")
    print(f"Profit (lucro líquido): {gm['net_profit']}")
    print(f"Brier Score: {sm['brier_score']}")
    print(f"Log Loss: {sm['log_loss']}")
    print(f"Calibration Error (ECE): {sm['calibration_error']}")
    print(f"Max Drawdown: {gm['max_drawdown']} ({gm['max_drawdown_pct']}%)")

    output_dir = args.output_dir
    csv_paths = report.to_csv(output_dir)
    print(f"\nCSV exportado para: {output_dir} ({len(csv_paths)} ficheiros)")

    if not args.no_markdown:
        md_path = report.to_markdown(f"{output_dir}/report.md")
        print(f"Markdown exportado para: {md_path}")

    if not args.no_excel:
        try:
            excel_path = report.to_excel(f"{output_dir}/report.xlsx")
            print(f"Excel exportado para: {excel_path}")
        except ImportError as exc:
            print(f"Excel não exportado: {exc}")

    plots_dir = None
    if not args.no_plots:
        plots_dir = f"{output_dir}/plots"
        try:
            plot_paths = report.generate_all_plots(plots_dir)
            print(f"Gráficos exportados para: {plots_dir} ({len(plot_paths)} ficheiros)")
        except ImportError as exc:
            print(f"Gráficos não exportados: {exc}")
            plots_dir = None

    if not args.no_html:
        html_path = report.to_html(f"{output_dir}/report.html", plots_dir=plots_dir)
        print(f"Relatório HTML exportado para: {html_path}")


if __name__ == "__main__":
    main()
