#!/usr/bin/env python3
"""
CLI para executar o Backtesting Framework histórico
(`src.backtest.historical`) sobre um dataset de jogos históricos (CSV).

Não altera nenhum algoritmo de previsão — Poisson, Dixon-Coles, Monte
Carlo, Goal Engine, Machine Learning, Kelly, Edge e EV permanecem
exatamente como estão. Este script apenas carrega jogos históricos
(`src.backtest.historical.dataset`), filtra-os conforme pedido e corre o
`BacktestEngine` já existente sobre eles.

Exemplos:

    # Backtest completo sobre todo o dataset
    python run_backtest.py --input jogos.csv

    # Backtest por intervalo de datas
    python run_backtest.py --input jogos.csv --start-date 2015-01-01 --end-date 2020-12-31

    # Backtest por competição
    python run_backtest.py --input jogos.csv --competition "Premier League"

    # Backtest por mercado
    python run_backtest.py --input jogos.csv --market OVER_2.5

    # Demo rápida com um pequeno conjunto de jogos históricos REAIS incluído
    # no repositório (examples/backtest/sample_real_games.csv)
    python run_backtest.py --demo
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtest.historical import BacktestEngine, FlatStake, KellyStake
from src.backtest.historical.dataset import filter_dataset, load_historical_dataset

DEMO_DATASET_PATH = ROOT_DIR / "examples" / "backtest" / "sample_real_games.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o Backtesting Framework histórico sobre um dataset de jogos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=str, default=None, help="Caminho para o CSV de jogos históricos.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help=f"Usa o pequeno dataset de jogos históricos reais incluído em {DEMO_DATASET_PATH}.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Filtra jogos a partir desta data (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Filtra jogos até esta data (YYYY-MM-DD).")
    parser.add_argument("--competition", type=str, default=None, help="Filtra por competição (ex. 'Premier League').")
    parser.add_argument("--market", type=str, default=None, help="Filtra por mercado (ex. 'OVER_2.5').")
    parser.add_argument(
        "--output-dir", type=str, default="output/backtest",
        help="Diretório onde gravar o relatório (CSV, Excel, HTML e gráficos).",
    )
    parser.add_argument(
        "--staking", choices=["flat", "kelly"], default="flat",
        help="Estratégia de stake (dimensiona a aposta; não altera Edge/EV/Kelly).",
    )
    parser.add_argument("--unit", type=float, default=1.0, help="Stake fixo, quando --staking=flat.")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Fração de Kelly, quando --staking=kelly.")
    parser.add_argument("--kelly-cap", type=float, default=0.05, help="Stake máximo (fração da banca), quando --staking=kelly.")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Banca de referência, quando --staking=kelly.")
    parser.add_argument(
        "--min-edge", type=float, default=3.0,
        help="Edge mínimo (%%) usado apenas para preencher 'engine_decision' em falta no dataset.",
    )
    parser.add_argument(
        "--max-kelly-fraction", type=float, default=0.25,
        help="Fração de Kelly máxima usada apenas para preencher 'engine_decision' em falta no dataset.",
    )
    parser.add_argument("--no-excel", action="store_true", help="Não gera o ficheiro Excel.")
    parser.add_argument("--no-plots", action="store_true", help="Não gera os gráficos (nem os embute no HTML).")
    parser.add_argument("--no-html", action="store_true", help="Não gera o relatório HTML.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input and not args.demo:
        parser.error("É necessário indicar --input <csv> ou usar --demo.")

    input_path = str(DEMO_DATASET_PATH) if args.demo else args.input

    print(f"A carregar dataset histórico de: {input_path}")
    dataset = load_historical_dataset(
        input_path,
        max_kelly_fraction=args.max_kelly_fraction,
        min_edge=args.min_edge,
    )
    print(f"Jogos carregados: {len(dataset)}")

    dataset = filter_dataset(
        dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        competition=args.competition,
        market=args.market,
    )
    print(f"Jogos após filtragem: {len(dataset)}")

    if dataset.empty:
        print("Nenhum jogo corresponde aos filtros indicados.", file=sys.stderr)
        sys.exit(1)

    if args.staking == "kelly":
        staking = KellyStake(fraction=args.kelly_fraction, cap=args.kelly_cap, bankroll=args.bankroll)
    else:
        staking = FlatStake(unit=args.unit)

    engine = BacktestEngine(staking=staking)
    report = engine.run(dataset)

    report.print_summary()

    output_dir = args.output_dir
    csv_paths = report.to_csv(output_dir)
    print(f"\nCSV exportado para: {output_dir} ({len(csv_paths)} ficheiros)")

    if not args.no_excel:
        try:
            excel_path = report.to_excel(f"{output_dir}/backtest_report.xlsx")
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
        html_path = report.to_html(f"{output_dir}/backtest_report.html", plots_dir=plots_dir)
        print(f"Relatório HTML exportado para: {html_path}")


if __name__ == "__main__":
    main()
