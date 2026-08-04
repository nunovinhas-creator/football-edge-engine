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

Não estima ataque/defesa por MLE sobre o histórico completo da liga. O que
este módulo faz é usar TODA a granularidade dos dados de confrontos
diretos (H2H) já devolvidos pela API, em vez de apenas a média agregada,
com encolhimento estatístico (`shrinkage`) para um prior quando a amostra
é pequena.

Melhoria #5 (auditoria matemática): esse prior deixou de ser sempre a
mesma constante fixa de liga. Quando `head_to_head` traz uma força de
ataque/defesa por equipa — calculada pelo Historical Dataset Builder já
existente (`src/engine/team_strength.py`, injetada por
`src.historical_dataset.backtest_bridge.derive_h2h`, nunca por este
módulo nem por uma API nova) — o encolhimento passa a ser feito para
essa força por equipa em vez do prior fixo (ver `_resolve_dynamic_prior`
abaixo). A força por equipa é, ela própria, primeiro encolhida para o
mesmo prior fixo de sempre, conforme a sua amostra — por isso o
comportamento anterior (prior fixo puro) é preservado exatamente quando
essa informação não está disponível (ver `docs/05_lambda_estimator.md`,
secção "Limitations", agora superada por este módulo).

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
# nesse módulo. Serve de prior de ÚLTIMO RECURSO (Nível 3) para o
# encolhimento estatístico abaixo — usado tal e qual quando não há força
# por equipa disponível (ver `_resolve_dynamic_prior`), e como o próprio
# prior para o qual a força por equipa é encolhida quando essa amostra é
# pequena (ver `src/engine/team_strength.py`).
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


def _resolve_dynamic_prior(h2h: Dict[str, Any]) -> Tuple[float, float]:
    """
    Nível 0 da cascata (Melhoria #5 da auditoria matemática — ver
    `src/engine/team_strength.py` e `docs/05_lambda_estimator.md`):
    resolve o prior usado no encolhimento (`_shrink_to_prior`) em
    `estimate_lambda_detailed`, no lugar do prior fixo de liga sempre que
    houver uma força por equipa disponível.

    `h2h` pode trazer, opcionalmente, as chaves `team_strength_home_goals`
    / `team_strength_away_goals` / `team_strength_sample_size` —
    preenchidas por `src.historical_dataset.backtest_bridge.derive_h2h` a
    partir do Historical Dataset Builder, nunca calculadas por este
    módulo (que continua sem qualquer dependência de `team_strength.py`,
    para preservar o seu contrato público e evitar import circular).

    Quando presentes, a força por equipa é ela própria encolhida para o
    prior fixo de liga, conforme a sua amostra (`_shrink_to_prior`, a
    mesma função, sem fórmula nova) — uma força vinda de 1-2 jogos
    históricos da equipa não deve substituir o prior fixo tão
    abruptamente como uma força vinda de dezenas de jogos substituiria.
    O resultado é o novo prior "estabilizado" (Nível 0) para o qual o
    H2H desta cascata (Nível 1) é encolhido (Nível 2).

    Quando ausentes, ausentes parcialmente, ou inválidas (negativas) —
    todo o código e todos os chamadores anteriores a esta melhoria, e
    qualquer chamador que não passe por `derive_h2h` — devolve
    exatamente `(LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS)`, o
    mesmo prior fixo de sempre (Nível 3), preservando 100% o
    comportamento existente.
    """
    home_strength = _safe_float(h2h.get("team_strength_home_goals"))
    away_strength = _safe_float(h2h.get("team_strength_away_goals"))
    if (
        home_strength is None
        or away_strength is None
        or home_strength < 0
        or away_strength < 0
    ):
        return LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS

    strength_sample_size = _safe_float(h2h.get("team_strength_sample_size")) or 0.0

    prior_home = _shrink_to_prior(home_strength, strength_sample_size, LEAGUE_PRIOR_HOME_GOALS)
    prior_away = _shrink_to_prior(away_strength, strength_sample_size, LEAGUE_PRIOR_AWAY_GOALS)
    return prior_home, prior_away


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

    prior_home_goals, prior_away_goals = _resolve_dynamic_prior(h2h)
    shrunk_home = _shrink_to_prior(raw_home_avg, sample_size, prior_home_goals)
    shrunk_away = _shrink_to_prior(raw_away_avg, sample_size, prior_away_goals)

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


# --------------------------------------------------------------------------
# Melhoria #8 (auditoria matemática): rótulo de confiança para o Framework
# de Avaliação (`src.evaluation`, `src.historical_dataset.backtest_bridge`).
# --------------------------------------------------------------------------
# `classify_model_confidence` NÃO é usada por nenhuma fórmula do motor
# (Dixon-Coles, Monte Carlo, Kelly, Edge, EV, Decision Engine) nem altera
# `lambda_home`/`mu_away` — é só uma categorização de
# `LambdaEstimate.tier` + `.effective_sample_size`, calculada depois da
# estimativa já estar pronta, usada exclusivamente para segmentar
# resultados de backtest por nível de confiança real do modelo (ver
# `src.evaluation.segments.segment_by_model_confidence`). Reutiliza o
# mesmo `SHRINKAGE_K` já definido acima como referência de "amostra
# suficiente para dominar o prior" — não introduz nenhuma constante nova
# sem justificação.
_STRONG_TIERS = ("recent_matches", "h2h_goal_totals")


def classify_model_confidence(tier: str, effective_sample_size: float) -> str:
    """
    Classifica a confiança do modelo em "HIGH" / "MEDIUM" / "LOW",
    combinando o nível de informação usado na cascata (`tier`) com a
    dimensão de amostra efetiva que a sustenta (`effective_sample_size`):

        - "avg_total_goals_or_prior" (Nível C/D, sem repartição empírica
          por equipa observada) nunca é classificado como "HIGH".
        - amostra efetiva >= 2*SHRINKAGE_K: domina claramente o prior no
          encolhimento (`_shrink_to_prior`) -- "HIGH" se também vier de um
          tier forte, "MEDIUM" caso contrário.
        - amostra efetiva >= SHRINKAGE_K: prior e amostra pesam
          aproximadamente o mesmo -- "MEDIUM".
        - amostra efetiva >= SHRINKAGE_K/2 e tier forte -- "MEDIUM".
        - abaixo disso: o prior de liga ainda domina a estimativa -- "LOW".

    Nunca lança exceção (entradas inválidas/negativas são tratadas como
    amostra zero). Uso exclusivo do Framework de Avaliação.
    """
    try:
        sample = float(effective_sample_size)
    except (TypeError, ValueError):
        sample = 0.0
    if sample != sample or sample < 0:  # NaN ou negativo
        sample = 0.0

    strong_tier = tier in _STRONG_TIERS

    if sample >= 2 * SHRINKAGE_K and strong_tier:
        return "HIGH"
    if sample >= SHRINKAGE_K:
        return "MEDIUM"
    if sample >= SHRINKAGE_K / 2 and strong_tier:
        return "MEDIUM"
    return "LOW"
