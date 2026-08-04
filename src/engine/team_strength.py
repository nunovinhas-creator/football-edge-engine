"""
Forças de ataque e defesa por equipa — Melhoria #5 da auditoria matemática.

Objetivo (ver `docs/AUDIT_MATEMATICA.md`, `docs/05_lambda_estimator.md`
§6 "Limitations"): `lambda_estimator.py` encolhia (`_shrink_to_prior`)
qualquer estimativa pobre em amostra sempre para o MESMO prior fixo de
liga (`LEAGUE_PRIOR_HOME_GOALS`/`LEAGUE_PRIOR_AWAY_GOALS`, uma constante
global), mesmo quando o Historical Dataset Builder (`src/historical_dataset/`)
já tem, no seu dataset normalizado, jogos suficientes de cada equipa para
produzir um prior mais informativo do que essa constante.

Este módulo calcula esse prior por equipa — golos marcados (ataque) e
sofridos (defesa), ponderados por recência — a partir exclusivamente do
dataset já produzido pelo Historical Dataset Builder (mesmo formato de
`src/historical_dataset/normalizer.py::NORMALIZED_COLUMNS`: `home_team`,
`away_team`, `home_score`, `away_score`, `date`). Não chama nenhuma API
nova nem lê nenhuma fonte de dados nova.

Não inventa fórmulas novas — reutiliza:
  - `apply_exponential_decay` (`src/engine/decay.py`), a mesma função já
    usada por `lambda_estimator.py` para `head_to_head.recent_matches`;
  - a mesma taxa de decaimento `RECENT_MATCH_DECAY_RATE` e a mesma amostra
    efetiva `_exponential_decay_effective_n` já definidas em
    `lambda_estimator.py` (importadas daqui, não duplicadas);
  - `_safe_float`, já usado em `lambda_estimator.py` para leitura
    defensiva de valores potencialmente `None`/`NaN`/inválidos.

A combinação golos-esperados = média(ataque da equipa, defesa do
adversário) é a repartição elementar já implícita em qualquer modelo de
golos casa/fora — não introduz um modelo estatístico novo.

Este módulo é importado por `src/historical_dataset/backtest_bridge.py`
(que injeta o resultado em `head_to_head`, sob chaves opcionais que
`lambda_estimator.py` já sabe ler) — NÃO por `lambda_estimator.py`, para
evitar import circular: `lambda_estimator.py` continua sem qualquer
dependência deste módulo, preservando o seu contrato público inalterado.

Sem fuga de informação (leakage): `compute_team_scoring_strength` só usa
jogos com `date` estritamente anterior a `before` — o mesmo padrão já
usado por `backtest_bridge.py::derive_h2h`.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.engine.decay import apply_exponential_decay
from src.engine.lambda_estimator import (
    RECENT_MATCH_DECAY_RATE,
    _exponential_decay_effective_n,
    _safe_float,
)


@dataclass(frozen=True)
class TeamStrength:
    """
    Ataque/defesa de uma equipa: golos médios marcados (`attack_avg`) e
    sofridos (`defense_avg`), ponderados por recência, e a amostra efetiva
    de decaimento (`sample_size`, mesmo conceito de
    `lambda_estimator._exponential_decay_effective_n`) que os sustenta.
    """

    attack_avg: float
    defense_avg: float
    sample_size: float


def _matches_before(df: pd.DataFrame, team: str, before: Any) -> pd.DataFrame:
    """Jogos de `team` (casa ou fora) com `date` estritamente anterior a `before`, ordenados cronologicamente."""
    if before is None or (hasattr(pd, "isna") and pd.isna(before)):
        return df.iloc[0:0]
    required = {"home_team", "away_team", "date"}
    if not required.issubset(df.columns):
        return df.iloc[0:0]

    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & df["date"].notna()
        & (df["date"] < before)
    )
    return df.loc[mask].sort_values("date")


def compute_team_scoring_strength(
    df: pd.DataFrame,
    team: str,
    before: Any,
    decay_rate: float = RECENT_MATCH_DECAY_RATE,
) -> Optional[TeamStrength]:
    """
    Golos marcados (ataque) e sofridos (defesa) por `team`, ponderados por
    recência (`apply_exponential_decay`), usando apenas jogos de `df`
    (dataset normalizado do Historical Dataset Builder) anteriores a
    `before` — sem fuga de informação.

    Devolve `None` quando não há nenhum jogo utilizável (equipa ausente do
    dataset, sem golos válidos, ou sem jogos anteriores a `before`) — o
    chamador (`estimate_team_strength_priors`) cai então para o próximo
    nível de informação disponível (H2H / prior fixo de liga), sem
    inventar um valor.
    """
    matches = _matches_before(df, team, before)
    if matches.empty:
        return None

    scored = []
    conceded = []
    for _, row in matches.iterrows():
        home_score = _safe_float(row.get("home_score"))
        away_score = _safe_float(row.get("away_score"))
        if home_score is None or away_score is None:
            continue
        if home_score < 0 or away_score < 0:
            continue
        if row["home_team"] == team:
            scored.append(home_score)
            conceded.append(away_score)
        else:
            scored.append(away_score)
            conceded.append(home_score)

    if not scored:
        return None

    attack_avg = float(apply_exponential_decay(np.array(scored, dtype=float), decay_rate=decay_rate))
    defense_avg = float(apply_exponential_decay(np.array(conceded, dtype=float), decay_rate=decay_rate))
    sample_size = _exponential_decay_effective_n(len(scored), decay_rate)

    return TeamStrength(attack_avg=attack_avg, defense_avg=defense_avg, sample_size=sample_size)


def estimate_team_strength_priors(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before: Any,
    decay_rate: float = RECENT_MATCH_DECAY_RATE,
) -> Optional[Dict[str, float]]:
    """
    Prior dinâmico por equipa para o confronto `home_team` x `away_team`,
    no formato de chaves opcionais que
    `src.engine.lambda_estimator._resolve_dynamic_prior` já sabe ler de
    `head_to_head` (Nível 0 da cascata — ver `docs/05_lambda_estimator.md`):
    `team_strength_home_goals`, `team_strength_away_goals`,
    `team_strength_sample_size`.

    Golos esperados de uma equipa = média entre o seu próprio ataque
    (`compute_team_scoring_strength`) e a defesa do adversário — a
    repartição ataque/defesa elementar, não um modelo novo.

    Devolve `None` quando uma das duas equipas não tem qualquer jogo
    anterior a `before` no dataset: não há base suficiente para uma força
    por equipa e o chamador deve deixar `lambda_estimator` cair para o
    prior fixo de liga (Nível 3).
    """
    home_strength = compute_team_scoring_strength(df, home_team, before, decay_rate)
    away_strength = compute_team_scoring_strength(df, away_team, before, decay_rate)
    if home_strength is None or away_strength is None:
        return None

    home_goals = (home_strength.attack_avg + away_strength.defense_avg) / 2.0
    away_goals = (away_strength.attack_avg + home_strength.defense_avg) / 2.0
    # A amostra que sustenta AMBOS os lados da combinação é limitada pela
    # equipa menos observada — nem o ataque de casa nem a defesa de fora
    # (e vice-versa) podem ser mais confiáveis do que o elo mais fraco.
    sample_size = min(home_strength.sample_size, away_strength.sample_size)

    return {
        "team_strength_home_goals": home_goals,
        "team_strength_away_goals": away_goals,
        "team_strength_sample_size": sample_size,
    }
