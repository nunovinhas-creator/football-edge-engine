"""Exportação do dataset histórico normalizado para CSV, SQLite e Parquet.

`export_all(...)` é o ponto de entrada principal: escreve sempre CSV e
SQLite (dependências já existentes no projeto — `pandas` e `sqlite3`, da
biblioteca padrão) e tenta escrever Parquet apenas se um motor Parquet
(`pyarrow` ou `fastparquet`) estiver instalado, devolvendo qual dos três
formatos foi efetivamente escrito (requisito "Parquet (se suportado)").
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd

from src.historical_dataset.normalizer import NORMALIZED_COLUMNS

logger = logging.getLogger(__name__)

RecordsLike = Union[pd.DataFrame, Iterable[Dict[str, Any]]]

DEFAULT_TABLE_NAME = "historical_matches"


def to_dataframe(records: RecordsLike) -> pd.DataFrame:
    """Normaliza `records` (DataFrame, lista de dicts ou generator) num DataFrame."""
    if isinstance(records, pd.DataFrame):
        return records.copy()

    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return pd.DataFrame(rows)


def to_csv(records: RecordsLike, path: Union[str, Path]) -> Path:
    """Exporta para CSV (uma linha por jogo)."""
    df = to_dataframe(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def to_sqlite(records: RecordsLike, path: Union[str, Path], table: str = DEFAULT_TABLE_NAME) -> Path:
    """Exporta para uma base de dados SQLite (tabela `table`, substituída se já existir)."""
    df = to_dataframe(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    return path


def to_parquet(records: RecordsLike, path: Union[str, Path]) -> Optional[Path]:
    """
    Exporta para Parquet, se um motor estiver instalado (`pyarrow` ou
    `fastparquet`). Devolve `None` (sem levantar exceção) se nenhum
    estiver disponível.
    """
    df = to_dataframe(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        logger.warning(
            "Parquet não suportado neste ambiente (nem pyarrow nem fastparquet "
            "instalados) — a ignorar exportação Parquet."
        )
        return None
    return path


def export_all(
    records: RecordsLike,
    output_dir: Union[str, Path],
    base_name: str = "historical_dataset",
    table: str = DEFAULT_TABLE_NAME,
) -> Dict[str, Optional[Path]]:
    """
    Exporta o dataset para CSV + SQLite + Parquet (se suportado) em
    `output_dir/{base_name}.{csv,sqlite,parquet}`. Devolve um dict com o
    caminho de cada formato efetivamente escrito (`None` para Parquet
    quando não suportado neste ambiente).
    """
    df = to_dataframe(records)
    output_dir = Path(output_dir)

    csv_path = to_csv(df, output_dir / f"{base_name}.csv")
    sqlite_path = to_sqlite(df, output_dir / f"{base_name}.sqlite", table=table)
    parquet_path = to_parquet(df, output_dir / f"{base_name}.parquet")

    return {"csv": csv_path, "sqlite": sqlite_path, "parquet": parquet_path}
