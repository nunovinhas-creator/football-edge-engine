"""Relatório de qualidade/execução do dataset histórico (`dataset_report.json`).

Módulo puramente de MEDIÇÃO sobre o dataset já construído/exportado pelo
Historical Dataset Builder — não calcula nem altera nenhuma odd, resultado,
probabilidade ou estatística; apenas resume o que já está no dataset
normalizado (`normalizer.NORMALIZED_COLUMNS`) e o que o `build_historical_dataset.py`
já sabe sobre a própria execução (tempo decorrido, nº de pedidos HTTP,
ficheiros exportados), para efeitos de auditoria de qualidade e diagnóstico.

Nenhuma fórmula do motor (Dixon-Coles, Monte Carlo, λ estimator, Kelly,
Edge, EV, Goal Engine, Machine Learning, Decision Engine) é usada aqui.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.historical_dataset.normalizer import NORMALIZED_COLUMNS
from src.historical_dataset.storage import RecordsLike, to_dataframe

ODDS_COLUMNS = [c for c in NORMALIZED_COLUMNS if c.startswith("odds_")]


def _missing_values_report(df: pd.DataFrame) -> Dict[str, Any]:
    """% de valores em falta, globalmente e por coluna."""
    if df.empty:
        return {"overall_pct": 0.0, "by_column": {}}

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    overall_pct = round((missing_cells / total_cells) * 100, 2) if total_cells else 0.0
    by_column = {col: round((df[col].isna().sum() / len(df)) * 100, 2) for col in df.columns}
    return {"overall_pct": overall_pct, "by_column": by_column}


def _count_games_with_odds(df: pd.DataFrame) -> int:
    """Nº de jogos com pelo menos uma odd (1X2/Over-Under/BTTS) publicada."""
    present = [c for c in ODDS_COLUMNS if c in df.columns]
    if not present:
        return 0
    return int(df[present].notna().any(axis=1).sum())


def _count_duplicate_games(df: pd.DataFrame) -> int:
    """Nº de linhas com `event_id` repetido (deveria ser 0 — builder/checkpoint já deduplicam)."""
    if "event_id" not in df.columns or df.empty:
        return 0
    return int(df["event_id"].duplicated().sum())


def _count_duplicate_odds(df: pd.DataFrame) -> int:
    """
    Nº de linhas, entre as que têm pelo menos uma odd publicada, cujas
    equipas/data/odds coincidem exatamente com outra linha do dataset —
    sinal de o mesmo jogo ter sido registado sob `event_id`s diferentes
    (ex. reprocessamento fora do alcance do checkpoint/dedup por chave
    `event_id`), distinto de `duplicate_games` (que compara só `event_id`).
    """
    present_odds = [c for c in ODDS_COLUMNS if c in df.columns]
    key_columns = [c for c in ("home_team", "away_team", "date") if c in df.columns] + present_odds
    if not present_odds or df.empty:
        return 0

    has_odds = df[present_odds].notna().any(axis=1)
    with_odds = df.loc[has_odds, key_columns]
    if with_odds.empty:
        return 0
    return int(with_odds.duplicated().sum())


def build_dataset_report(
    records: RecordsLike,
    *,
    competition: Optional[Any] = None,
    season: Optional[Any] = None,
    execution_time_seconds: Optional[float] = None,
    api_requests: Optional[int] = None,
    output_files: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
) -> Dict[str, Any]:
    """
    Constrói o relatório de qualidade/execução para o dataset já obtido
    (`records`: DataFrame, lista de dicts ou generator já materializado —
    ver `storage.to_dataframe`).

    `competition`/`season`: o que foi pedido nesta execução (ex.
    `--competition-id`/`--season-id` do CLI), ou `None`/"all" quando não
    filtrado — passados tal como recebidos, sem qualquer transformação.

    `execution_time_seconds`/`api_requests`: medidos pelo chamador (CLI) —
    este módulo não mede tempo nem conta pedidos HTTP, apenas reporta o que
    lhe é passado.

    `output_files`: mapa formato -> caminho efetivamente escrito (ou
    `None`/omitido para formatos não gerados nesta execução — ex.
    `--output csv` só escreve `csv`).
    """
    df = to_dataframe(records)

    return {
        "competition": competition,
        "season": season,
        "total_games": int(len(df)),
        "total_odds": _count_games_with_odds(df),
        "duplicate_games": _count_duplicate_games(df),
        "duplicate_odds": _count_duplicate_odds(df),
        "missing_values": _missing_values_report(df),
        "execution_time_seconds": execution_time_seconds,
        "api_requests": api_requests,
        "output_files": {k: (str(v) if v else None) for k, v in (output_files or {}).items()},
    }


def write_dataset_report(report: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Grava `report` como JSON legível (`indent=2`) em `path`, criando o diretório se necessário."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
