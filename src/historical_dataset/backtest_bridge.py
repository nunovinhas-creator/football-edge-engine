"""Ponte entre o Historical Dataset Builder e o Backtesting Framework já existente.

Converte o dataset normalizado (um registo por jogo, produzido por
`builder.HistoricalDatasetBuilder`) num DataFrame no formato aceite por
`src.backtest.historical.dataset.load_historical_dataset`, para um
mercado específico (ex. "HOME", "OVER_2.5", "BTTS").

Este módulo NÃO calcula nem inventa `model_prob` — a probabilidade
prevista pelo modelo tem de vir de quem chama (tipicamente o output do
motor de previsão já existente, `src.engine.*`, aplicado a estes jogos),
porque este builder não invoca nem altera nenhuma fórmula matemática do
projeto (Dixon-Coles, Monte Carlo, Kelly, Edge, EV, Goal Engine, Machine
Learning). Ver `docs/07_historical_dataset_builder.md`.
"""

from typing import Any, Callable, Optional, Union

import pandas as pd

from src.historical_dataset.storage import to_dataframe

ValueSource = Union[str, Callable[[pd.Series], Any], pd.Series, list, tuple, float, int]

# Mapeia um mercado (mesma nomenclatura usada por src.backtest.historical e
# src.engine.decision) para a coluna de odds correspondente no dataset
# normalizado (ver normalizer.NORMALIZED_COLUMNS).
MARKET_ODDS_COLUMN = {
    "HOME": "odds_home",
    "DRAW": "odds_draw",
    "AWAY": "odds_away",
    "OVER_1.5": "odds_over_1_5",
    "UNDER_1.5": "odds_under_1_5",
    "OVER_2.5": "odds_over_2_5",
    "UNDER_2.5": "odds_under_2_5",
    "OVER_3.5": "odds_over_3_5",
    "UNDER_3.5": "odds_under_3_5",
    "BTTS": "odds_btts_yes",
    "BTTS_YES": "odds_btts_yes",
    "BTTS_NO": "odds_btts_no",
}

BACKTEST_BRIDGE_COLUMNS = [
    "date", "competition", "home_team", "away_team", "market",
    "odd", "model_prob", "home_goals", "away_goals",
]


def _resolve(source: Optional[ValueSource], df: pd.DataFrame) -> Optional[pd.Series]:
    """Resolve um valor por linha a partir de escalar, nome de coluna, callable ou Series/lista."""
    if source is None:
        return None
    if callable(source):
        return df.apply(source, axis=1)
    if isinstance(source, str) and source in df.columns:
        return df[source]
    if isinstance(source, (pd.Series, list, tuple)):
        return pd.Series(list(source), index=df.index)
    return pd.Series([source] * len(df), index=df.index)


def to_backtest_frame(
    records,
    market: str,
    model_prob: ValueSource,
    engine_decision: Optional[ValueSource] = None,
    result: Optional[ValueSource] = None,
) -> pd.DataFrame:
    """
    Converte o dataset normalizado num DataFrame pronto para
    `src.backtest.historical.dataset.load_historical_dataset`, para o
    mercado `market`.

    `model_prob` é obrigatório e nunca é calculado aqui: passe a
    probabilidade já prevista pelo motor de previsão existente para estes
    jogos — uma Series/lista alinhada com `records`, o nome de uma coluna
    já presente em `records`, um valor único, ou uma função `row -> float`.

    `engine_decision` e `result`, se omitidos, ficam de fora do DataFrame
    devolvido: `load_historical_dataset` já os preenche automaticamente a
    partir de `home_goals`/`away_goals` e da odd/probabilidade fornecidas,
    sem que este módulo tenha de repetir essa lógica.
    """
    df = to_dataframe(records)
    if df.empty:
        return pd.DataFrame(columns=BACKTEST_BRIDGE_COLUMNS)

    market_norm = str(market).strip().upper()
    odds_column = MARKET_ODDS_COLUMN.get(market_norm)
    if odds_column is None:
        raise ValueError(
            f"Mercado não suportado pela ponte de backtest: {market!r}. "
            f"Mercados suportados: {sorted(MARKET_ODDS_COLUMN)}"
        )
    if odds_column not in df.columns:
        raise KeyError(f"Coluna de odds em falta no dataset: {odds_column!r}")

    out = pd.DataFrame({
        "date": df["date"],
        "competition": df["competition"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "market": market_norm,
        "odd": df[odds_column],
        "home_goals": df["home_score"],
        "away_goals": df["away_score"],
    })

    out["model_prob"] = _resolve(model_prob, df)

    resolved_decision = _resolve(engine_decision, df)
    if resolved_decision is not None:
        out["engine_decision"] = resolved_decision

    resolved_result = _resolve(result, df)
    if resolved_result is not None:
        out["result"] = resolved_result

    # Jogos sem odd publicada para este mercado (ou sem resultado final
    # conhecido) não podem ser avaliados; descarta-los aqui em vez de
    # deixar `load_historical_dataset` falhar a converter `None` para float.
    out = out.dropna(subset=["odd", "home_goals", "away_goals"]).reset_index(drop=True)

    return out
