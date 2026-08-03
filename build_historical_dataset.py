#!/usr/bin/env python3
"""
CLI para correr o Historical Dataset Builder (`src.historical_dataset`) de
ponta a ponta: competições -> épocas -> jogos terminados -> odds ->
estatísticas -> dataset normalizado -> CSV/SQLite/Parquet.

Não altera nenhum algoritmo de previsão — Poisson, Dixon-Coles, Monte
Carlo, Goal Engine, Machine Learning, Kelly, Edge e EV permanecem
exatamente como estão. Este script apenas percorre a BSD API e normaliza
os dados brutos (ver `docs/07_historical_dataset_builder.md`).

Exemplos:

    # Todas as ligas ativas, sem checkpoint (execução única)
    python build_historical_dataset.py --output-dir data/historical

    # Apenas duas ligas específicas, com checkpoint/resume
    python build_historical_dataset.py --leagues 39,140 --checkpoint-dir data/historical/.checkpoint

    # Execução parcial, limitada a 200 jogos (ex. para testar rapidamente)
    python build_historical_dataset.py --max-events 200
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.historical_dataset.builder import HistoricalDatasetBuilder
from src.historical_dataset.checkpoint import Checkpoint, NullCheckpoint
from src.historical_dataset.client import BSDHistoricalClient
from src.historical_dataset.rate_limiter import RateLimiter
from src.historical_dataset.storage import export_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Constrói um dataset histórico real a partir da BSD API "
            "(competições -> épocas -> jogos terminados -> odds -> estatísticas)."
        )
    )
    parser.add_argument("--output-dir", default="data/historical", help="Diretório de saída (CSV/SQLite/Parquet).")
    parser.add_argument("--base-name", default="historical_dataset", help="Nome base dos ficheiros exportados.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Diretório de checkpoint/resume (omitir para uma execução única, sem retomar).",
    )
    parser.add_argument(
        "--leagues",
        default=None,
        help="IDs de liga separados por vírgula (omitir para todas as ligas ativas).",
    )
    parser.add_argument("--country", default=None, help="Filtrar ligas por país (ignorado se --leagues for usado).")
    parser.add_argument("--include-inactive", action="store_true", help="Incluir ligas inativas na listagem.")
    parser.add_argument("--page-size", type=int, default=100, help="Tamanho de página nos pedidos paginados (máx. 200).")
    parser.add_argument("--rate-limit", type=float, default=5.0, help="Máximo de pedidos por segundo à BSD API.")
    parser.add_argument("--max-events", type=int, default=None, help="Limite de jogos processados nesta execução.")
    parser.add_argument("--no-odds", action="store_true", help="Não obter odds por jogo.")
    parser.add_argument("--no-stats", action="store_true", help="Não obter estatísticas por jogo.")
    parser.add_argument(
        "--odds-comparison",
        action="store_true",
        help="Também obter a comparação de bookmakers por jogo (/odds/comparison/).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    client = BSDHistoricalClient(
        rate_limiter=RateLimiter(max_calls=max(1, round(args.rate_limit)), period_seconds=1.0)
    )
    checkpoint = Checkpoint(args.checkpoint_dir) if args.checkpoint_dir else NullCheckpoint()

    builder = HistoricalDatasetBuilder(
        client=client,
        checkpoint=checkpoint,
        page_size=args.page_size,
        include_odds=not args.no_odds,
        include_stats=not args.no_stats,
        include_odds_comparison=args.odds_comparison,
    )

    leagues = None
    if args.leagues:
        leagues = [int(x) for x in args.leagues.split(",") if x.strip()]
    elif args.country or args.include_inactive:
        leagues = list(
            builder.iter_competitions(country=args.country, include_inactive=args.include_inactive)
        )

    print("A construir dataset histórico a partir da BSD API...")
    records = []
    try:
        for i, record in enumerate(builder.build(leagues=leagues, max_events=args.max_events), start=1):
            records.append(record)
            if i % 50 == 0:
                print(f"  {i} jogos processados...")
    finally:
        checkpoint.close()

    print(f"Total de jogos processados: {len(records)}")

    if not records:
        print("Nenhum jogo obtido — nada para exportar.")
        return

    paths = export_all(records, args.output_dir, base_name=args.base_name)
    print("Ficheiros exportados:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path if path else '(não suportado neste ambiente)'}")


if __name__ == "__main__":
    main()
