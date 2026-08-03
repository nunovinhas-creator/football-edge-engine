"""
Carregamento de jogos históricos para o Backtesting Framework.

Este módulo é responsável apenas por CARREGAR e NORMALIZAR jogos
históricos (a partir de CSV, de um `DataFrame` já em memória, ou de uma
lista de dicts) num formato que o `BacktestEngine` já aceita — nunca
recalcula probabilidade, Edge, EV ou Kelly, que continuam a ser
calculados exclusivamente por `evaluator.py` a partir de
`src.engine.edge` / `src.engine.kelly`.

Duas operações de "preenchimento" são feitas aqui, porque decorrem
diretamente dos dados brutos (não de uma previsão nova):

    1. `engine_decision` em falta é preenchida chamando o `DecisionEngine`
       REAL (`src.engine.decision`) sobre a probabilidade do modelo e a
       odd já presentes no dataset — ou seja, reutiliza a decisão que o
       motor de previsão já produziria para esses valores, não inventa
       uma nova regra.
    2. `result` (resultado final do mercado) em falta é derivado do
       resultado final do jogo (golos casa/fora) através de uma tabela de
       mapeamento mercado -> resultado (`infer_market_result`), que é pura
       aritmética sobre o resultado já ocorrido — não uma previsão.

O projeto não tem (ainda) uma integração própria com uma fonte de jogos
históricos com odds e resultados; por isso a via suportada é CSV (aceita
também DataFrame/lista de dicts para quem já tiver os dados carregados de
outra forma).
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from src.engine.decision import DecisionEngine

PathLike = Union[str, Path]
DatasetSource = Union[PathLike, pd.DataFrame, Iterable[Dict[str, Any]]]

# Campos que o dataset de jogos históricos tem de conter, no mínimo, antes
# de qualquer preenchimento automático (ver módulo `docstring` acima):
# data, competição, equipa da casa, equipa visitante, mercado recomendado,
# odd disponível e probabilidade prevista pelo motor. `result`/`resultado`
# pode ser derivado de `home_goals`/`away_goals` quando ausente.
REQUIRED_COLUMNS_MIN = [
    "date",
    "competition",
    "home_team",
    "away_team",
    "market",
    "odd",
    "model_prob",
]

# Aceita as mesmas chaves em português e inglês usadas por
# `HistoricalBet.from_dict`, mais os campos específicos deste módulo
# (golos casa/fora, usados para derivar o resultado do mercado).
_ALIASES = {
    "date": ("date", "data"),
    "competition": ("competition", "competicao", "liga", "league"),
    "home_team": ("home_team", "equipa_casa", "casa", "home"),
    "away_team": ("away_team", "equipa_visitante", "visitante", "fora", "away"),
    "market": ("market", "mercado"),
    "odd": ("odd", "odd_disponivel", "available_odd", "bookie_odd"),
    "model_prob": ("model_prob", "probabilidade", "probabilidade_prevista", "prob_model"),
    "engine_decision": ("engine_decision", "decisao", "decisao_motor", "decision"),
    "result": ("result", "resultado", "resultado_final", "resultado_mercado"),
    "home_goals": ("home_goals", "golos_casa", "home_score", "golos_equipa_casa"),
    "away_goals": ("away_goals", "golos_fora", "away_score", "golos_equipa_visitante"),
}

_OVER_UNDER_RE = re.compile(r"^(OVER|UNDER)[_ ]?(\d+(?:\.\d+)?)$", re.IGNORECASE)
_BTTS_RE = re.compile(r"^BTTS[_ ]?(YES|NO)?$", re.IGNORECASE)


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas em português (ou aliases em inglês) para o nome canónico."""
    rename_map = {}
    for canonical, aliases in _ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def infer_market_result(market: str, home_goals: float, away_goals: float) -> str:
    """
    Deriva o resultado final ("WIN"/"LOSS") de um mercado a partir do
    resultado final do jogo (golos casa/fora).

    É apenas uma tabela de mapeamento mercado -> resultado sobre um
    resultado já ocorrido; não prevê nada e não envolve probabilidade,
    Edge, EV ou Kelly. Suporta os mercados usados no resto do projeto:
    HOME/DRAW/AWAY, OVER_X.X/UNDER_X.X e BTTS (ambas marcam / não marcam).
    """
    market_norm = str(market).strip().upper()
    home_goals = float(home_goals)
    away_goals = float(away_goals)
    total_goals = home_goals + away_goals

    if market_norm in ("HOME", "1", "CASA"):
        return "WIN" if home_goals > away_goals else "LOSS"
    if market_norm in ("AWAY", "2", "FORA"):
        return "WIN" if away_goals > home_goals else "LOSS"
    if market_norm in ("DRAW", "X", "EMPATE"):
        return "WIN" if home_goals == away_goals else "LOSS"

    over_under_match = _OVER_UNDER_RE.match(market_norm)
    if over_under_match:
        direction, threshold_str = over_under_match.groups()
        threshold = float(threshold_str)
        if direction.upper() == "OVER":
            return "WIN" if total_goals > threshold else "LOSS"
        return "WIN" if total_goals < threshold else "LOSS"

    btts_match = _BTTS_RE.match(market_norm)
    if btts_match:
        both_scored = home_goals > 0 and away_goals > 0
        variant = (btts_match.group(1) or "YES").upper()
        return "WIN" if (both_scored if variant == "YES" else not both_scored) else "LOSS"

    raise ValueError(
        f"Não é possível derivar automaticamente o resultado do mercado {market!r} "
        "a partir do resultado final (golos casa/fora). Forneça a coluna "
        "'result'/'resultado' diretamente para este mercado."
    )


def _fill_missing_results(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche `result` em falta a partir de `home_goals`/`away_goals` (ver `infer_market_result`)."""
    if "result" in df.columns and df["result"].notna().all():
        return df

    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        if "result" not in df.columns:
            raise KeyError(
                "O dataset precisa de 'result'/'resultado' OU de "
                "'home_goals'+'away_goals' ('golos_casa'+'golos_fora') "
                "para determinar o resultado final do mercado."
            )
        return df

    df = df.copy()
    if "result" not in df.columns:
        df["result"] = None
    missing = df["result"].isna()
    df.loc[missing, "result"] = df.loc[missing].apply(
        lambda row: infer_market_result(row["market"], row["home_goals"], row["away_goals"]),
        axis=1,
    )
    return df


def _fill_missing_decisions(
    df: pd.DataFrame,
    max_kelly_fraction: float = 0.25,
    min_edge: float = 3.0,
) -> pd.DataFrame:
    """
    Preenche `engine_decision` em falta chamando o `DecisionEngine` real
    (`src.engine.decision.DecisionEngine`) sobre a probabilidade do modelo
    e a odd já presentes no dataset — reutiliza a decisão que o motor de
    previsão já produziria para esses valores, não recalcula edge/EV/Kelly
    com uma fórmula própria.
    """
    if "engine_decision" in df.columns and df["engine_decision"].notna().all():
        return df

    df = df.copy()
    if "engine_decision" not in df.columns:
        df["engine_decision"] = None
    engine = DecisionEngine(max_kelly_fraction=max_kelly_fraction, min_edge=min_edge)
    missing = df["engine_decision"].isna()
    df.loc[missing, "engine_decision"] = df.loc[missing].apply(
        lambda row: engine.evaluate_bet(row["market"], float(row["model_prob"]) * 100.0, float(row["odd"])).action,
        axis=1,
    )
    return df


def _build_match_column(df: pd.DataFrame) -> pd.DataFrame:
    if "match" in df.columns and df["match"].notna().all():
        return df
    df = df.copy()
    df["match"] = df["home_team"].astype(str) + " vs " + df["away_team"].astype(str)
    return df


def load_games_from_csv(path: PathLike, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Lê um CSV de jogos históricos tal qual (sem normalizar). Ver `load_historical_dataset`."""
    return pd.read_csv(path, **read_csv_kwargs)


def _to_dataframe(source: DatasetSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return load_games_from_csv(source)
    rows: List[Dict[str, Any]] = list(source)
    return pd.DataFrame(rows)


def load_historical_dataset(
    source: DatasetSource,
    max_kelly_fraction: float = 0.25,
    min_edge: float = 3.0,
) -> pd.DataFrame:
    """
    Ponto de entrada único do módulo: carrega jogos históricos a partir de
    um caminho CSV, de um `DataFrame` ou de uma lista de dicts, e devolve
    um `DataFrame` normalizado pronto para `BacktestEngine.run(...)`.

    Garante que cada registo contém, no mínimo: data do jogo, competição,
    equipa da casa, equipa visitante, mercado recomendado, odd disponível
    no momento da previsão e probabilidade prevista pelo motor. Edge, EV e
    Kelly NÃO são calculados aqui — são calculados depois pelo
    `BacktestEngine`/`evaluator.py` a partir da odd e da probabilidade
    aqui presentes, reutilizando `src.engine.edge` / `src.engine.kelly`.

    `engine_decision` e `result` (resultado final do mercado) são
    preenchidos automaticamente quando ausentes (ver `_fill_missing_decisions`
    e `_fill_missing_results`); se já vierem no dataset, são respeitados
    tal como fornecidos.
    """
    df = _to_dataframe(source)
    if df.empty:
        return df

    df = _rename_known_columns(df)

    missing_required = [c for c in REQUIRED_COLUMNS_MIN if c not in df.columns]
    if missing_required:
        raise KeyError(f"Colunas obrigatórias em falta no dataset histórico: {missing_required}")

    df["odd"] = df["odd"].astype(float)
    df["model_prob"] = df["model_prob"].astype(float)

    df = _fill_missing_results(df)
    df = _fill_missing_decisions(df, max_kelly_fraction=max_kelly_fraction, min_edge=min_edge)
    df = _build_match_column(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def filter_dataset(
    df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    competition: Optional[str] = None,
    market: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filtra um dataset já normalizado (ver `load_historical_dataset`) por
    intervalo de datas, competição e/ou mercado. Usado pelo CLI
    `run_backtest.py` para os modos "por intervalo de datas", "por
    competição" e "por mercado" (nenhum filtro = backtest completo).
    """
    if df.empty:
        return df

    working = df.copy()
    if "date" in working.columns and not pd.api.types.is_datetime64_any_dtype(working["date"]):
        working["date"] = pd.to_datetime(working["date"], errors="coerce")

    if start_date is not None:
        working = working[working["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        working = working[working["date"] <= pd.to_datetime(end_date)]
    if competition is not None:
        working = working[working["competition"].astype(str).str.casefold() == str(competition).casefold()]
    if market is not None:
        working = working[working["market"].astype(str).str.casefold() == str(market).casefold()]

    return working.reset_index(drop=True)
