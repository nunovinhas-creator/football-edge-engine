"""
Exemplo de execução do Backtesting Framework histórico
(src/backtest/historical).

Gera um conjunto de jogos históricos sintéticos (ver
`src.backtest.historical.sample_data`), corre o `BacktestEngine` sobre
eles e imprime/exporta um relatório completo: tabela resumo, CSV,
Excel, e gráficos (evolução da banca, distribuições, curva de
calibração).

Uso:
    python -m src.tools.run_backtest_example [output_dir] [--n-games N]

Nota: os dados usados aqui são inteiramente SINTÉTICOS (o repositório não
contém um dataset histórico real). Servem apenas para demonstrar o
funcionamento do framework; para validar o modelo real, substituir
`generate_sample_dataset(...)` por dados reais carregados de CSV/DB
através de `BacktestEngine.run(...)` (aceita lista de dicts ou DataFrame).
"""

import argparse
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.backtest.historical import BacktestEngine, FlatStake, generate_sample_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", nargs="?", default="output/backtest_example",
        help="Diretório onde gravar o relatório (CSV/Excel/gráficos).",
    )
    parser.add_argument("--n-games", type=int, default=500, help="Número de jogos sintéticos a gerar.")
    parser.add_argument("--seed", type=int, default=42, help="Seed do gerador aleatório (reprodutibilidade).")
    args = parser.parse_args()

    print(f"A gerar {args.n_games} apostas históricas sintéticas (seed={args.seed})...")
    dataset = generate_sample_dataset(n_games=args.n_games, seed=args.seed)

    engine = BacktestEngine(staking=FlatStake(unit=1.0))
    report = engine.run(dataset)

    report.print_summary()

    print("\n--- Threshold Analysis (Edge) ---")
    print(report.edge_thresholds.to_string(index=False))

    print("\n--- Segmentos disponíveis ---")
    for name, segment_df in report.segments.items():
        print(f"\n[{name}]")
        print(segment_df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_paths = report.to_csv(args.output_dir)
    print(f"\nCSV exportado para: {args.output_dir} ({len(csv_paths)} ficheiros)")

    try:
        excel_path = report.to_excel(os.path.join(args.output_dir, "backtest_report.xlsx"))
        print(f"Excel exportado para: {excel_path}")
    except ImportError as exc:
        print(f"Excel não exportado ({exc})")

    plots_dir = os.path.join(args.output_dir, "plots")
    plot_paths = report.generate_all_plots(plots_dir)
    print(f"Gráficos exportados para: {plots_dir} ({len(plot_paths)} ficheiros)")


if __name__ == "__main__":
    main()
