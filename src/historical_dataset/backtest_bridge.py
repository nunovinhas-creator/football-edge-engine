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

from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from src.engine.lambda_estimator import (
    classify_model_confidence,
    estimate_lambda,
    estimate_lambda_detailed,
)
from src.engine.team_strength import estimate_team_strength_priors
from src.engine.value import estimate_pregame_probabilities
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


def _naive_dates(dates: pd.Series) -> pd.Series:
    """
    Normaliza `event_date` (ISO 8601 UTC, ex. `2024-08-10T15:00:00Z`, tal
    como a BSD API devolve — ver `docs/07_historical_dataset_builder.md`,
    "Limitações", ponto 7) para datetime SEM fuso horário.

    Incompatibilidade de esquema pura, não uma alteração de dados: o
    `sample_real_games.csv` ilustrativo usado por `run_backtest.py --demo`
    só tem datas simples (`2011-10-23`), por isso este caso nunca tinha
    sido exercitado antes — mas `openpyxl`/`pandas.ExcelWriter`
    (`BacktestReport.to_excel`/`EvaluationReport.to_excel`, já existentes)
    rejeitam datetimes com fuso horário. Converter para UTC "naive" aqui
    preserva o instante exato (só remove a anotação de fuso, já que tudo
    já está em UTC) sem tocar em `load_historical_dataset` nem em nenhum
    exportador.
    """
    parsed = pd.to_datetime(dates, errors="coerce", utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def to_backtest_frame(
    records,
    market: str,
    model_prob: ValueSource,
    engine_decision: Optional[ValueSource] = None,
    result: Optional[ValueSource] = None,
    model_confidence: Optional[ValueSource] = None,
    lambda_tier: Optional[ValueSource] = None,
    effective_sample_size: Optional[ValueSource] = None,
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

    `model_confidence`, `lambda_tier` e `effective_sample_size` (Melhoria
    #8 da auditoria matemática): OPCIONAIS, mesmo mecanismo de resolução
    que `model_prob` — tipicamente alimentados por
    `lambda_confidence_from_dixon_coles(records)`. Se omitidos, ficam de
    fora do DataFrame devolvido (retrocompatível: `HistoricalBet` já trata
    a sua ausência sem erro).
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
        "date": _naive_dates(df["date"]),
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

    resolved_model_confidence = _resolve(model_confidence, df)
    if resolved_model_confidence is not None:
        out["model_confidence"] = resolved_model_confidence

    resolved_lambda_tier = _resolve(lambda_tier, df)
    if resolved_lambda_tier is not None:
        out["lambda_tier"] = resolved_lambda_tier

    resolved_effective_sample_size = _resolve(effective_sample_size, df)
    if resolved_effective_sample_size is not None:
        out["effective_sample_size"] = resolved_effective_sample_size

    # Jogos sem odd publicada para este mercado (ou sem resultado final
    # conhecido) não podem ser avaliados; descarta-los aqui em vez de
    # deixar `load_historical_dataset` falhar a converter `None` para float.
    out = out.dropna(subset=["odd", "home_goals", "away_goals"]).reset_index(drop=True)

    return out


def _reorient_score(row: pd.Series, home_team: str, away_team: str) -> "tuple[float, float]":
    """Golos de `row` (um jogo passado) na perspetiva do jogo A JOGAR (`home_team`/`away_team`).

    `estimate_lambda`/`estimate_pregame_lambdas` assumem `head_to_head`
    orientado à identidade das equipas do próximo jogo, não ao mando de
    campo de cada confronto passado (ver `docs/05_lambda_estimator.md`,
    secção "Assumptions", ponto 1) — por isso, quando `row` teve `home_team`
    a jogar fora (ou vice-versa), os golos são trocados aqui.
    """
    if row["home_team"] == home_team:
        return float(row["home_score"]), float(row["away_score"])
    return float(row["away_score"]), float(row["home_score"])


def derive_h2h(df: pd.DataFrame, home_team: str, away_team: str, before: Any) -> Dict[str, Any]:
    """
    Deriva um dict `head_to_head` — no mesmo formato que
    `src.engine.lambda_estimator.estimate_lambda` /
    `src.engine.pregame_lambda.estimate_pregame_lambdas` já esperam de
    `EventCollector.get_matches()` para jogos futuros — a partir de
    confrontos diretos ANTERIORES entre `home_team` e `away_team` já
    presentes no dataset do Historical Dataset Builder.

    Não faz nenhum pedido adicional à BSD API (o Historical Dataset
    Builder não devolve `head_to_head` por jogo) e não tem fuga de
    informação: `before` é sempre a data do jogo a avaliar, e só entram
    confrontos (ou jogos de cada equipa, ver abaixo) com data estritamente
    anterior. Sem confrontos diretos anteriores nem força por equipa
    calculável (equipas que nunca jogaram no dataset antes de `before`, ou
    jogo sem data), devolve `{}` — `estimate_lambda({})` já trata esse
    caso de forma defensiva (cai para o prior de liga), sem alterações
    aqui.

    Melhoria #5 (auditoria matemática): além dos confrontos diretos entre
    `home_team` e `away_team`, também tenta calcular a força de
    ataque/defesa de CADA equipa (`src.engine.team_strength`, Nível 0 da
    cascata de `lambda_estimator.py`) a partir de TODOS os jogos
    anteriores de cada equipa já no dataset (não só os confrontos diretos
    entre estas duas) — sem fuga de informação (mesmo corte por `before`).
    Isto fica disponível mesmo quando as duas equipas nunca se defrontaram
    diretamente, exatamente o caso em que o estimador só tinha, até agora,
    o prior fixo de liga como alternativa.
    """
    if before is None or pd.isna(before):
        return {}

    h2h: Dict[str, Any] = {}

    pair_mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    prior = df.loc[pair_mask & df["date"].notna() & (df["date"] < before)].sort_values("date")

    if not prior.empty:
        recent_matches = []
        home_goals_total = 0.0
        away_goals_total = 0.0
        home_wins = 0
        away_wins = 0
        for _, row in prior.iterrows():
            h_goals, a_goals = _reorient_score(row, home_team, away_team)
            match_date = row["date"]
            recent_matches.append({
                "home_goals": h_goals,
                "away_goals": a_goals,
                "date": match_date.isoformat() if hasattr(match_date, "isoformat") else str(match_date),
            })
            home_goals_total += h_goals
            away_goals_total += a_goals
            if h_goals > a_goals:
                home_wins += 1
            elif a_goals > h_goals:
                away_wins += 1

        total_matches = len(recent_matches)
        h2h.update({
            "total_matches": total_matches,
            "home_goals": home_goals_total,
            "away_goals": away_goals_total,
            "avg_total_goals": (home_goals_total + away_goals_total) / total_matches,
            "home_win_rate": 100.0 * home_wins / total_matches,
            "away_win_rate": 100.0 * away_wins / total_matches,
            "recent_matches": recent_matches,
        })

    team_strength_priors = estimate_team_strength_priors(df, home_team, away_team, before)
    if team_strength_priors is not None:
        h2h.update(team_strength_priors)

    return h2h


def model_probabilities_from_dixon_coles(records) -> Dict[Any, Dict[str, float]]:
    """
    Para cada jogo do dataset normalizado, calcula a probabilidade 1X2
    (fração) do modelo Dixon-Coles já em produção — exatamente os mesmos
    módulos que `src.collector.client.EventCollector.get_matches()` já usa
    para jogos futuros (`src.engine.lambda_estimator.estimate_lambda` +
    `src.engine.value.estimate_pregame_probabilities`, ver
    `docs/AUDIT_MATEMATICA.md` §15/§16) — aplicados aqui a jogos já
    terminados, com `head_to_head` derivado de confrontos diretos
    anteriores dentro do próprio dataset (`derive_h2h`, sem fuga de
    informação).

    Não recalcula nem altera nenhuma fórmula do Dixon-Coles, do estimador
    de lambda, nem do adaptador `pregame_lambda`: esta função só prepara o
    input (`head_to_head`) que essas funções já esperavam e que, ao
    contrário de jogos futuros pedidos à BSD API, este builder não
    devolve por jogo.

    Devolve `{event_id: {"home": p, "draw": p, "away": p}}`.
    """
    df = to_dataframe(records)
    if df.empty:
        return {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    probabilities: Dict[Any, Dict[str, float]] = {}
    for _, row in df.iterrows():
        h2h = derive_h2h(df, row["home_team"], row["away_team"], row["date"])
        lambda_home, mu_away = estimate_lambda(h2h)
        probabilities[row["event_id"]] = estimate_pregame_probabilities(lambda_home, mu_away)
    return probabilities


def lambda_confidence_from_dixon_coles(records) -> Dict[Any, Dict[str, Any]]:
    """
    Melhoria #8 (auditoria matemática): propaga a confiança do estimador de
    lambda (`src.engine.lambda_estimator.LambdaEstimate.tier` e
    `.effective_sample_size`) até ao Evaluation Framework, para cada jogo
    do dataset normalizado — mesmo `head_to_head` derivado por `derive_h2h`
    usado por `model_probabilities_from_dixon_coles` (nenhuma fuga de
    informação nova), mas devolvendo a PROVENIÊNCIA da estimativa em vez
    do par (lambda_home, mu_away).

    Não recalcula nem substitui o Dixon-Coles, o estimador de lambda, nem
    `model_probabilities_from_dixon_coles`: esta função só chama
    `estimate_lambda_detailed` (em vez de `estimate_lambda`) sobre o mesmo
    input, para obter os campos de proveniência que `estimate_lambda` já
    descartava. `model_prob` continua a vir exclusivamente de
    `model_probabilities_from_dixon_coles` — o resultado desta função serve
    apenas como METADADO opcional de avaliação (ver
    `to_backtest_frame(..., lambda_tier=..., effective_sample_size=...,
    model_confidence=...)`), nunca como input de Kelly/Edge/EV/Decision
    Engine.

    Devolve `{event_id: {"lambda_tier": str, "effective_sample_size": float,
    "model_confidence": str}}`.
    """
    df = to_dataframe(records)
    if df.empty:
        return {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    confidence: Dict[Any, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        h2h = derive_h2h(df, row["home_team"], row["away_team"], row["date"])
        estimate = estimate_lambda_detailed(h2h)
        confidence[row["event_id"]] = {
            "lambda_tier": estimate.tier,
            "effective_sample_size": estimate.effective_sample_size,
            "model_confidence": classify_model_confidence(
                estimate.tier, estimate.effective_sample_size
            ),
        }
    return confidence
