"""
Estimador pré-jogo estatisticamente mais forte para `lambda_home`/`mu_away`
(inputs de `dixon_coles_simulate_match()`, `src/engine/dixon_coles.py`).

Contexto: `src/engine/pregame_lambda.py::estimate_pregame_lambdas()` (mantido
sem alterações, ver esse ficheiro) reparte `head_to_head.avg_total_goals`
por uma partilha fixa (`home_share`) ajustada por um tilt limitado de
win-rate. É um adaptador correto mas deliberadamente simples (ver
`docs/AUDIT_MATEMATICA.md`, §3.2/§12.2): não usa a granularidade dos dados
de H2H já devolvidos pela API (golos por equipa, jogos recentes), nem
pondera a confiança na amostra disponível.

Este módulo substitui esse adaptador como estimador por omissão, mantendo
o mesmo contrato (`estimate_lambda(h2h) -> (lambda_home, mu_away)`, nunca
lança exceção, nunca devolve <= 0) mas usando MAIS da informação já
presente no mesmo objeto `head_to_head` (ver `schema.yaml`, campo
`head_to_head` do endpoint de eventos: `total_matches`, `home_wins`,
`draws`, `away_wins`, `home_goals`, `away_goals`, `avg_total_goals`,
`home_win_rate`, `away_win_rate`, `recent_matches`) e reutilizando
`src/engine/decay.py::apply_exponential_decay` — uma função já existente
no repositório para ponderação exponencial de séries temporais, referida
em `docs/AUDIT_MATEMATICA.md` §12.2 como "pronta mas não ligada ao
Dixon-Coles".

Não estima ataque/defesa por MLE sobre o histórico completo da liga (isso
exigiria uma fonte de dados que o projeto não recolhe hoje — ver
`docs/05_lambda_estimator.md`, secção "Limitations"). O que este módulo
faz é usar TODA a granularidade dos dados de confrontos diretos (H2H) já
devolvidos pela API, em vez de apenas a média agregada, com encolhimento
estatístico (`shrinkage`) para a média de liga quando a amostra é pequena.

O Dixon-Coles em si (`tau`, `dixon_coles_simulate_match`,
`calculate_fractional_kelly`) não é tocado por este módulo.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.engine.decay import apply_exponential_decay
from src.engine.pregame_lambda import (
    DEFAULT_AVG_TOTAL_GOALS,
    HOME_ADVANTAGE_SHARE,
    MIN_LAMBDA,
    estimate_pregame_lambdas as estimate_pregame_lambdas_heuristic,
)

# --------------------------------------------------------------------------
# Prior de liga
# --------------------------------------------------------------------------
# Não deriva de um novo número inventado: é o mesmo DEFAULT_AVG_TOTAL_GOALS
# (2.5 golos/jogo) já usado por `pregame_lambda.py` quando não há H2H,
# repartido pela mesma vantagem de casa (`HOME_ADVANTAGE_SHARE`) já assumida
# nesse módulo. Serve de "prior" para o encolhimento estatístico abaixo — o
# projeto não tem, hoje, uma fonte de média de liga dinâmica por competição
# (ver `docs/05_lambda_estimator.md`, secção "Limitations").
LEAGUE_PRIOR_HOME_GOALS = round(DEFAULT_AVG_TOTAL_GOALS * (0.5 + HOME_ADVANTAGE_SHARE), 3)
LEAGUE_PRIOR_AWAY_GOALS = round(DEFAULT_AVG_TOTAL_GOALS * (0.5 - HOME_ADVANTAGE_SHARE), 3)

# Cap superior de segurança numérica. dixon_coles_simulate_match() trunca a
# matriz de resultados em max_goals=8 (src/engine/dixon_coles.py); um lambda
# muito acima disto começaria a perder massa de probabilidade real para fora
# da matriz. Não existia um cap equivalente no heurístico anterior — é uma
# proteção adicional, não uma mudança de comportamento normal (só atua em
# cenários patológicos de input).
MAX_LAMBDA = 6.0

# Força do encolhimento (shrinkage) para a média de liga: número de
# "pseudo-jogos" de prior que uma observação precisa de superar antes de a
# amostra real dominar a estimativa. K mais alto = mais conservador com
# amostras pequenas. K=4 significa que, com 4 jogos observados, prior e
# dados reais pesam o mesmo; com poucos jogos (1-2), o prior ainda domina;
# com uma amostra grande (>=15-20), o efeito do prior torna-se residual.
SHRINKAGE_K = 4.0

# A repartição de golos por equipa em `avg_total_goals` (Nível C, ver
# `_split_from_avg_total_goals`) é inferida (via win-rate tilt), não
# observada diretamente por equipa — por isso a confiança nessa repartição
# é descontada face ao `total_matches` bruto, que mede a confiança na SOMA
# (avg_total_goals), não na forma como foi repartida.
TIER_C_CONFIDENCE_DISCOUNT = 0.5

# Ponderação exponencial de jogos recentes dentro de `recent_matches` (ver
# `_split_from_recent_matches`). Mesma ordem de grandeza do `decay_rate`
# por omissão de `apply_exponential_decay` (0.05) escalado para o facto de
# aqui a série ser "por jogo" (tipicamente 3-10 entradas), não "por rodada
# de liga" — um decaimento mais acentuado é necessário para que os últimos
# 2-3 confrontos pesem visivelmente mais do que os mais antigos.
RECENT_MATCH_DECAY_RATE = 0.35

# Nº mínimo de entradas válidas em `recent_matches` para preferir esta via
# à média agregada (`home_goals`/`away_goals`/`avg_total_goals`). Com 0-1
# jogos não há "recência" que ponderar — cai-se para o próximo nível.
MIN_RECENT_MATCHES_FOR_TIER_A = 2


@dataclass(frozen=True)
class LambdaEstimate:
    """
    Resultado detalhado do estimador — além do par (lambda_home, mu_away)
    que `dixon_coles_simulate_match()` exige, guarda a proveniência da
    estimativa para explicabilidade/depuração (princípio nº4 de
    `docs/01_project_scope.md`: "Every model must be explainable").
    """

    lambda_home: float
    mu_away: float
    tier: str  # "recent_matches" | "h2h_goal_totals" | "avg_total_goals_or_prior"
    effective_sample_size: float
    raw_home_goals_avg: float
    raw_away_goals_avg: float

    def as_tuple(self) -> Tuple[float, float]:
        return self.lambda_home, self.mu_away


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _shrink_to_prior(
    raw_value: float,
    sample_size: float,
    prior: float,
    k: float = SHRINKAGE_K,
) -> float:
    """
    Encolhimento estatístico (empirical-Bayes shrinkage) de uma média
    observada para um prior de liga, ponderado pela dimensão da amostra:

        shrunk = (raw_value * n + prior * k) / (n + k)

    n=0            -> shrunk == prior (sem dados, confia-se inteiramente no
                      prior de liga).
    n -> infinito  -> shrunk -> raw_value (amostra suficiente para dominar
                      o prior).
    n == k         -> prior e amostra pesam exatamente o mesmo.

    É a mesma lógica usada em modelos de taxas com amostras pequenas
    (ex. estimar a "verdadeira" média de golos de um confronto direto com
    apenas 2-3 jogos não deve produzir uma estimativa tão extrema como a
    média bruta desses 2-3 jogos sugeriria).
    """
    sample_size = max(0.0, sample_size)
    return (raw_value * sample_size + prior * k) / (sample_size + k)


def _exponential_decay_effective_n(n: int, decay_rate: float) -> float:
    """
    Dimensão de amostra EFETIVA (design effect, `1 / sum(weight_i^2)` sobre
    pesos normalizados) de uma média ponderada por decaimento exponencial
    sobre `n` observações.

    Mirror do vetor de pesos usado internamente por
    `apply_exponential_decay()` (`src/engine/decay.py`) — não substitui essa
    função (que continua a ser a única a calcular a MÉDIA ponderada usada
    em `_split_from_recent_matches`), serve apenas para responder a uma
    pergunta diferente: "quantas observações independentes esta média
    ponderada realmente vale, para efeitos de confiança/encolhimento?".

    Com pesos exponenciais, `n_eff` NÃO cresce indefinidamente com `n` —
    estabiliza perto de `(1+q)/(1-q)` (q=exp(-decay_rate)) porque jogos
    antigos contribuem cada vez menos. Isto é intencional: uma média
    fortemente enviesada para os últimos 3-4 jogos não deve ser tratada,
    para efeitos de `_shrink_to_prior`, com a mesma confiança que uma média
    simples sobre as mesmas `n` observações teria (ver
    `docs/05_lambda_estimator.md`, secção "Validation" — foi exatamente
    esta lacuna que o benchmark de recuperação sintética
    (`scripts/benchmark_lambda_estimator.py`) expôs antes desta correção).
    """
    if n <= 0:
        return 0.0
    weights = np.exp(-decay_rate * np.arange(n))[::-1]
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights ** 2))


def _chronological_order(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ordena `recent_matches` do mais antigo para o mais recente — a
    convenção exigida por `apply_exponential_decay` (índice 0 = mais
    antigo). Se todas as entradas tiverem "date", ordena por essa data.
    Caso contrário, assume-se a convenção REST mais comum (lista devolvida
    do mais recente para o mais antigo) e inverte-se a ordem — assunção
    documentada em `docs/05_lambda_estimator.md` (o projeto não tem, no
    momento desta alteração, acesso a uma API key para confirmar
    empiricamente a ordem devolvida por `head_to_head.recent_matches`).
    """
    if entries and all(entry.get("date") for entry in entries):
        return sorted(entries, key=lambda entry: entry["date"])
    return list(reversed(entries))


def _split_from_recent_matches(h2h: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Nível A (melhor informação disponível): repartição de golos casa/fora
    ponderada por recência a partir de `head_to_head.recent_matches`.

    Cada entrada é um jogo direto anterior entre as duas equipas; aceita-se
    "home_goals"/"away_goals" (mesma convenção de nomes já usada no dict
    agregado `head_to_head`), interpretados de forma consistente com
    `home_win_rate`/`away_win_rate` já usados por `predict_probability()` e
    `pregame_lambda.py`: orientados à equipa da casa/fora do jogo A JOGAR,
    não ao mando de campo de cada confronto passado individualmente (ver
    `docs/05_lambda_estimator.md`, secção "Assumptions").

    Devolve None se `recent_matches` estiver ausente, não for uma lista, ou
    tiver menos de `MIN_RECENT_MATCHES_FOR_TIER_A` entradas válidas — nesse
    caso o chamador cai para o Nível B.
    """
    raw_entries = h2h.get("recent_matches")
    if not isinstance(raw_entries, list) or not raw_entries:
        return None

    usable = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        home_goals = _safe_float(entry.get("home_goals"))
        away_goals = _safe_float(entry.get("away_goals"))
        if home_goals is None or away_goals is None:
            continue
        if home_goals < 0 or away_goals < 0:
            continue
        usable.append(entry)

    if len(usable) < MIN_RECENT_MATCHES_FOR_TIER_A:
        return None

    ordered = _chronological_order(usable)
    home_series = np.array([_safe_float(e["home_goals"]) for e in ordered], dtype=float)
    away_series = np.array([_safe_float(e["away_goals"]) for e in ordered], dtype=float)

    home_avg = float(apply_exponential_decay(home_series, decay_rate=RECENT_MATCH_DECAY_RATE))
    away_avg = float(apply_exponential_decay(away_series, decay_rate=RECENT_MATCH_DECAY_RATE))

    # Amostra efetiva, não a contagem bruta de jogos — ver
    # `_exponential_decay_effective_n`. Evita que o encolhimento confie
    # tanto numa média fortemente concentrada nos últimos 3-4 confrontos
    # como confiaria numa média simples sobre o mesmo número de jogos.
    effective_n = _exponential_decay_effective_n(len(usable), RECENT_MATCH_DECAY_RATE)

    return home_avg, away_avg, effective_n


def _split_from_h2h_goal_totals(h2h: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Nível B: repartição empírica direta a partir de `home_goals`/
    `away_goals`/`total_matches` agregados do H2H — golos médios realmente
    marcados por cada equipa nos confrontos diretos, em vez de inferidos a
    partir do total (`avg_total_goals`) e de uma inclinação de win-rate.
    """
    total_matches = _safe_float(h2h.get("total_matches"))
    home_goals_total = _safe_float(h2h.get("home_goals"))
    away_goals_total = _safe_float(h2h.get("away_goals"))

    if not total_matches or total_matches <= 0:
        return None
    if home_goals_total is None or away_goals_total is None:
        return None
    if home_goals_total < 0 or away_goals_total < 0:
        return None

    home_avg = home_goals_total / total_matches
    away_avg = away_goals_total / total_matches
    return home_avg, away_avg, total_matches


def _split_from_avg_total_goals_or_prior(h2h: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Nível C/D: sem dados suficientes para uma repartição empírica por
    equipa. Reutiliza o adaptador já existente e testado
    (`pregame_lambda.py::estimate_pregame_lambdas`) para obter uma
    repartição a partir de `avg_total_goals` (ou do prior de liga, se nem
    isso existir) — em vez de reimplementar a mesma lógica de tilt de
    win-rate/vantagem de casa neste módulo.

    A confiança nesta repartição é descontada (`TIER_C_CONFIDENCE_DISCOUNT`)
    porque `total_matches` mede a confiança na SOMA (avg_total_goals), não
    na forma como essa soma foi repartida entre as duas equipas.
    """
    home_avg, away_avg = estimate_pregame_lambdas_heuristic(h2h)
    total_matches = _safe_float(h2h.get("total_matches")) or 0.0
    sample_size = total_matches * TIER_C_CONFIDENCE_DISCOUNT
    return home_avg, away_avg, sample_size


def estimate_lambda_detailed(h2h: Optional[Dict[str, Any]]) -> LambdaEstimate:
    """
    Versão "explicável" de `estimate_lambda()` — devolve também a
    proveniência (`tier`) e os valores brutos antes do encolhimento, úteis
    para testes, depuração e para o script de benchmark
    (`scripts/benchmark_lambda_estimator.py`).

    Nunca lança exceção: qualquer input inesperado (None, {}, tipos
    inválidos, listas malformadas) degrada graciosamente para o próximo
    nível de informação disponível, terminando no prior de liga.
    """
    h2h = h2h or {}
    if not isinstance(h2h, dict):
        h2h = {}

    tier = "avg_total_goals_or_prior"
    split = _split_from_recent_matches(h2h)
    if split is not None:
        tier = "recent_matches"
    else:
        split = _split_from_h2h_goal_totals(h2h)
        if split is not None:
            tier = "h2h_goal_totals"
        else:
            split = _split_from_avg_total_goals_or_prior(h2h)

    raw_home_avg, raw_away_avg, sample_size = split

    shrunk_home = _shrink_to_prior(raw_home_avg, sample_size, LEAGUE_PRIOR_HOME_GOALS)
    shrunk_away = _shrink_to_prior(raw_away_avg, sample_size, LEAGUE_PRIOR_AWAY_GOALS)

    lambda_home = min(MAX_LAMBDA, max(MIN_LAMBDA, round(shrunk_home, 3)))
    mu_away = min(MAX_LAMBDA, max(MIN_LAMBDA, round(shrunk_away, 3)))

    return LambdaEstimate(
        lambda_home=lambda_home,
        mu_away=mu_away,
        tier=tier,
        effective_sample_size=sample_size,
        raw_home_goals_avg=round(raw_home_avg, 3),
        raw_away_goals_avg=round(raw_away_avg, 3),
    )


def estimate_lambda(h2h: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Ponto de entrada por omissão para a pipeline (`src/collector/client.py`):
    mesmo contrato de `pregame_lambda.py::estimate_pregame_lambdas(h2h)` —
    devolve sempre `(lambda_home, mu_away)`, floats > 0, nunca lança
    exceção — mas usando a estimativa estatisticamente mais forte descrita
    no docstring do módulo.
    """
    return estimate_lambda_detailed(h2h).as_tuple()
