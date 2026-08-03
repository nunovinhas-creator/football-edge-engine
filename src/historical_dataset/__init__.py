"""Historical Dataset Builder — pipeline de construção de um dataset histórico real via BSD API.

Percorre competições, épocas, jogos terminados, odds e estatísticas
disponíveis na BSD API e produz um dataset único normalizado (exportável
para CSV, SQLite e Parquet), sem calcular nem alterar nenhum modelo
matemático do projeto (Dixon-Coles, Monte Carlo, Kelly, Edge, EV, Goal
Engine, Machine Learning). Ver `docs/07_historical_dataset_builder.md`.
"""

from src.historical_dataset.backtest_bridge import to_backtest_frame
from src.historical_dataset.builder import HistoricalDatasetBuilder
from src.historical_dataset.checkpoint import Checkpoint, NullCheckpoint
from src.historical_dataset.client import BSDAPIError, BSDHistoricalClient
from src.historical_dataset.rate_limiter import RateLimiter
from src.historical_dataset.report import build_dataset_report, write_dataset_report
from src.historical_dataset.storage import export_all, to_csv, to_dataframe, to_parquet, to_sqlite

__all__ = [
    "HistoricalDatasetBuilder",
    "BSDHistoricalClient",
    "BSDAPIError",
    "RateLimiter",
    "Checkpoint",
    "NullCheckpoint",
    "to_backtest_frame",
    "export_all",
    "to_csv",
    "to_sqlite",
    "to_parquet",
    "to_dataframe",
    "build_dataset_report",
    "write_dataset_report",
]
