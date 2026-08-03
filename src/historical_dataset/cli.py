"""CLI oficial do Historical Dataset Builder (`src.historical_dataset`).

Percorre a BSD API (competições -> épocas -> jogos terminados -> odds ->
estatísticas), normaliza cada jogo e exporta o dataset resultante, com
logging de progresso e um relatório de qualidade/execução
(`dataset_report.json`, ver `report.py`).

Não calcula nem altera nenhum algoritmo de previsão — Poisson, Dixon-Coles,
Monte Carlo, Goal Engine, Machine Learning, Kelly, Edge e EV permanecem
exatamente como estão. Este módulo apenas orquestra a BSD API e formata
progresso/relatório em torno do `HistoricalDatasetBuilder` já existente.

**Segurança:** nada neste módulo imprime a chave de API, o header
`Authorization` ou qualquer token — os eventos de progresso do builder só
transportam IDs, nomes e contadores (ver `builder.HistoricalDatasetBuilder.build`),
nunca `client.api_key`.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.historical_dataset.builder import HistoricalDatasetBuilder
from src.historical_dataset.checkpoint import Checkpoint, NullCheckpoint
from src.historical_dataset.client import BSDHistoricalClient
from src.historical_dataset.rate_limiter import RateLimiter
from src.historical_dataset.report import build_dataset_report, write_dataset_report
from src.historical_dataset.storage import to_csv, to_dataframe, to_parquet, to_sqlite

OUTPUT_CHOICES = ("csv", "sqlite", "parquet", "all")
RESUME_CHOICES = ("true", "false")


class ProgressLogger:
    """
    Handler de progresso passado como `progress_callback` ao
    `HistoricalDatasetBuilder`: imprime competição, época, página, jogos
    processados, odds processadas e uma estimativa de ETA.

    Sem `--max-events`, o total de jogos a processar não é conhecido à
    partida (a BSD API não expõe uma contagem total), por isso o ETA fica
    "desconhecido" nesse caso — reportar um número fabricado seria
    enganoso. Com `--max-events`, o ETA é `(max_events - processados) /
    taxa_atual`.

    Só recebe/imprime os campos que o builder emite (IDs, nomes,
    contadores) — nunca `client.api_key` nem headers HTTP.
    """

    def __init__(self, max_events: Optional[int] = None, print_every: int = 25):
        self.max_events = max_events
        self.print_every = max(1, print_every)
        self._start = time.monotonic()

    def _eta_str(self, games_processed: int, elapsed: float) -> str:
        if self.max_events is None or games_processed <= 0 or elapsed <= 0:
            return "desconhecido"
        rate = games_processed / elapsed
        if rate <= 0:
            return "desconhecido"
        remaining = max(self.max_events - games_processed, 0)
        return f"{remaining / rate:.0f}s"

    def __call__(self, event_type: str, data: Dict[str, Any]) -> None:
        if event_type == "competition_start":
            print(f"[competição] {data.get('league_name') or '(sem nome)'} (id={data.get('league_id')})")
        elif event_type == "season_start":
            print(f"  [época] {data.get('season_name') or '(sem nome)'} (id={data.get('season_id')})")
        elif event_type == "page":
            print(f"    [página {data.get('page_number')}] {data.get('items_count')} jogos obtidos")
        elif event_type == "event":
            games = data.get("games_processed", 0)
            if games % self.print_every != 0:
                return
            elapsed = time.monotonic() - self._start
            rate = games / elapsed if elapsed > 0 else 0.0
            eta = self._eta_str(games, elapsed)
            print(
                f"    jogos processados: {games} | odds processadas: {data.get('odds_processed', 0)} "
                f"| {rate:.2f} jogos/s | ETA: {eta}"
            )
        elif event_type == "season_done":
            print(f"  [época concluída] id={data.get('season_id')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Constrói um dataset histórico real a partir da BSD API "
            "(competições -> épocas -> jogos terminados -> odds -> estatísticas)."
        )
    )
    parser.add_argument("--output-dir", default="data/historical", help="Diretório de saída.")
    parser.add_argument("--base-name", default="historical_dataset", help="Nome base dos ficheiros exportados.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Diretório de checkpoint/resume. Se omitido e --resume=true, usa <output-dir>/.checkpoint.",
    )
    parser.add_argument(
        "--leagues",
        default=None,
        help="IDs de liga separados por vírgula (omitir para todas as ligas ativas). Não combinar com --competition-id.",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        default=None,
        help="Atalho para uma única competição (equivalente a --leagues <id>). Não combinar com --leagues.",
    )
    parser.add_argument("--season-id", type=int, default=None, help="Restringe a construção a uma única época.")
    parser.add_argument("--country", default=None, help="Filtrar ligas por país (ignorado se --leagues/--competition-id forem usados).")
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
    parser.add_argument(
        "--output",
        choices=OUTPUT_CHOICES,
        default="all",
        help="Formato(s) a exportar: csv, sqlite, parquet ou all (todos).",
    )
    parser.add_argument(
        "--resume",
        choices=RESUME_CHOICES,
        default="false",
        help="Ativa checkpoint/resume desta execução (retoma épocas/jogos já concluídos num --checkpoint-dir anterior).",
    )
    return parser


def _resolve_leagues(args: argparse.Namespace, builder: HistoricalDatasetBuilder):
    if args.competition_id is not None:
        return [args.competition_id]
    if args.leagues:
        return [int(x) for x in args.leagues.split(",") if x.strip()]
    if args.country or args.include_inactive:
        return list(builder.iter_competitions(country=args.country, include_inactive=args.include_inactive))
    return None


def _export(records: list, output_dir: Path, base_name: str, output_format: str) -> Dict[str, Optional[str]]:
    """Exporta `records` nos formatos pedidos por `--output` (csv/sqlite/parquet/all)."""
    df = to_dataframe(records)
    output_files: Dict[str, Optional[str]] = {}

    if output_format in ("csv", "all"):
        output_files["csv"] = str(to_csv(df, output_dir / f"{base_name}.csv"))
    if output_format in ("sqlite", "all"):
        output_files["sqlite"] = str(to_sqlite(df, output_dir / f"{base_name}.sqlite"))
    if output_format in ("parquet", "all"):
        parquet_path = to_parquet(df, output_dir / f"{base_name}.parquet")
        output_files["parquet"] = str(parquet_path) if parquet_path else None

    return output_files


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.competition_id is not None and args.leagues:
        parser.error("--competition-id e --leagues não podem ser usados em conjunto.")

    resume = args.resume == "true"
    checkpoint_dir = args.checkpoint_dir or (str(Path(args.output_dir) / ".checkpoint") if resume else None)

    client = BSDHistoricalClient(
        rate_limiter=RateLimiter(max_calls=max(1, round(args.rate_limit)), period_seconds=1.0)
    )
    checkpoint = Checkpoint(checkpoint_dir) if resume else NullCheckpoint()
    progress = ProgressLogger(max_events=args.max_events)

    builder = HistoricalDatasetBuilder(
        client=client,
        checkpoint=checkpoint,
        page_size=args.page_size,
        include_odds=not args.no_odds,
        include_stats=not args.no_stats,
        include_odds_comparison=args.odds_comparison,
        progress_callback=progress,
    )

    leagues = _resolve_leagues(args, builder)
    season_ids = [args.season_id] if args.season_id is not None else None

    print("A construir dataset histórico a partir da BSD API...")
    start = time.monotonic()
    records = []
    try:
        for record in builder.build(leagues=leagues, max_events=args.max_events, season_ids=season_ids):
            records.append(record)
    finally:
        checkpoint.close()
    elapsed = time.monotonic() - start

    print(f"Total de jogos processados: {len(records)}")

    output_dir = Path(args.output_dir)
    output_files: Dict[str, Optional[str]] = {}
    if records:
        output_files = _export(records, output_dir, args.base_name, args.output)
        print("Ficheiros exportados:")
        for fmt, path in output_files.items():
            print(f"  {fmt}: {path if path else '(não suportado neste ambiente)'}")
    else:
        print("Nenhum jogo obtido — nada para exportar.")

    report = build_dataset_report(
        records,
        competition=args.competition_id if args.competition_id is not None else (args.leagues or "all"),
        season=args.season_id if args.season_id is not None else "all",
        execution_time_seconds=round(elapsed, 3),
        api_requests=client.request_count,
        output_files=output_files,
    )
    report_path = write_dataset_report(report, output_dir / "dataset_report.json")
    print(f"Relatório de qualidade exportado para: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
