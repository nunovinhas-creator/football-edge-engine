"""
Kelly Criterion — fração de banca a apostar.

Melhoria #6 (auditoria matemática, `docs/AUDIT_MATEMATICA.md` §7): antes
desta melhoria existiam três implementações independentes de Kelly
fracionário (`kelly.py`, `src.engine.dixon_coles.calculate_fractional_kelly`,
`src.engine.decision.DecisionEngine.evaluate_bet`), todas com a mesma
fórmula de Kelly completo mas a fração fixa (0.25 / 1/4 Kelly) hard-coded
de forma independente em cada uma. Este módulo passa a ser a fonte única
não só do Kelly completo (`kelly_fraction`), mas também de como essa fração
fixa é escalada pela confiança do modelo (`calculate_adaptive_kelly_fraction`)
— as outras implementações (`dixon_coles.py`, `decision.py`,
`backtest.historical.staking`) chamam estas duas funções em vez de
recalcular a fórmula.

A confiança do modelo reutiliza EXCLUSIVAMENTE `LambdaEstimate.tier` e
`LambdaEstimate.effective_sample_size` (já produzidos por
`src.engine.lambda_estimator.estimate_lambda_detailed`, Melhoria #5).
Nenhum indicador novo é introduzido, e nenhuma probabilidade/odd/edge/EV/
critério de seleção de aposta é tocado por este módulo — só a FRAÇÃO de
Kelly usada, ou seja, o tamanho do stake de uma aposta já decidida.

Nota de desenho: o multiplicador de confiança usa `lambda_tier` para
escolher a constante de saturação `k` (ver `_TIER_SATURATION_K` abaixo),
mas NÃO usa `classify_model_confidence` (Melhoria #8) para isso — essa
função tem fronteiras rígidas em `effective_sample_size` (ex. muda de
rótulo exatamente em `n_eff==8`), pelo que a usar aqui produziria um
"salto" descontínuo do multiplicador exatamente nessa fronteira (violando
o requisito de continuidade). `lambda_tier`, ao contrário de
`effective_sample_size`, é fixo para uma dada estimativa — usá-lo (só ele)
para escolher `k` mantém o multiplicador uma função contínua e suave de
`effective_sample_size` em toda a gama, para qualquer tier.
"""

from typing import Optional

from src.engine.lambda_estimator import SHRINKAGE_K

# Constante de saturação (`k` em `n_eff / (n_eff + k)`) por valor de
# `LambdaEstimate.tier` — reflete a mesma hierarquia de qualidade de
# informação já documentada em `lambda_estimator.py` (cascata Nível A
# `recent_matches` > Nível B `h2h_goal_totals` > Nível C/D
# `avg_total_goals_or_prior`, ver `estimate_lambda_detailed`): tiers
# melhores precisam de menos amostra efetiva para a mesma confiança na
# fração de stake (`k` menor => satura mais depressa para perto de 1).
# Reutiliza sempre o mesmo `SHRINKAGE_K` (=4.0) já definido em
# `lambda_estimator.py` como unidade de referência — só a escala por
# tier é nova, nenhuma constante de base é reinventada.
_TIER_SATURATION_K = {
    "recent_matches": SHRINKAGE_K,          # Nível A — melhor informação
    "h2h_goal_totals": SHRINKAGE_K * 2.0,   # Nível B
    "avg_total_goals_or_prior": SHRINKAGE_K * 4.0,  # Nível C/D — pior informação
}
# Tier desconhecido/inesperado: tratado com a mesma cautela do pior nível
# conhecido (nunca lança exceção, nunca assume o melhor caso por omissão).
_DEFAULT_TIER_SATURATION_K = _TIER_SATURATION_K["avg_total_goals_or_prior"]


def calculate_confidence_multiplier(
    lambda_tier: Optional[str] = None,
    effective_sample_size: Optional[float] = None,
) -> float:
    """
    Multiplicador de confiança (0.0-1.0) aplicado à fração base de Kelly.

    Depende exclusivamente de `lambda_tier` (`LambdaEstimate.tier`) e
    `effective_sample_size` (`LambdaEstimate.effective_sample_size`),
    combinados de forma contínua e limitada por:

        multiplier = n_eff / (n_eff + k)

    onde `k = _TIER_SATURATION_K[lambda_tier]` (ver acima). Esta forma
    (mesma família de `_shrink_to_prior` em `lambda_estimator.py`) é
    contínua e suave em `n_eff` para qualquer `lambda_tier` fixo (`k` não
    depende de `n_eff`), estritamente crescente, e limitada a [0, 1) —
    nunca escala a fração base para cima, só para baixo, conforme a
    confiança é menor que "amostra infinita".

    Retrocompatibilidade: se `lambda_tier` OU `effective_sample_size` não
    forem fornecidos (`None`, o valor por omissão), devolve exatamente
    `1.0` — sem qualquer escala — para que o comportamento seja idêntico
    ao existente antes desta melhoria. O mesmo acontece se
    `effective_sample_size` não for um número válido (nunca lança
    exceção).
    """
    if lambda_tier is None or effective_sample_size is None:
        return 1.0

    try:
        n_eff = float(effective_sample_size)
    except (TypeError, ValueError):
        return 1.0
    if n_eff != n_eff:  # NaN
        return 1.0
    n_eff = max(0.0, n_eff)

    k = _TIER_SATURATION_K.get(lambda_tier, _DEFAULT_TIER_SATURATION_K)

    return n_eff / (n_eff + k)


def calculate_adaptive_kelly_fraction(
    base_fraction: float = 0.25,
    lambda_tier: Optional[str] = None,
    effective_sample_size: Optional[float] = None,
) -> float:
    """
    Função única responsável por calcular a fração de Kelly adaptativa —
    todas as implementações de Kelly do projeto (`fractional_kelly` abaixo,
    `src.engine.dixon_coles.calculate_fractional_kelly`,
    `src.engine.decision.DecisionEngine.evaluate_bet`,
    `src.backtest.historical.staking.KellyStake`) chamam esta função em vez
    de escalar `base_fraction` pela confiança de forma independente.

        fraction = base_fraction * confidence_multiplier

    Sem `lambda_tier`/`effective_sample_size` (omissos), devolve
    exatamente `base_fraction` — retrocompatibilidade total com o
    comportamento anterior a esta melhoria (fração fixa, sem escala).
    """
    return base_fraction * calculate_confidence_multiplier(lambda_tier, effective_sample_size)


def kelly_fraction(probability, odd):
    """
    Kelly Criterion

    probability: probabilidade do modelo (0-1)
    odd: odd decimal

    retorna fração da banca
    """

    b = odd - 1
    p = probability
    q = 1 - p

    kelly = ((b * p) - q) / b

    return max(kelly, 0)


def fractional_kelly(
    probability,
    odd,
    fraction: float = 0.25,
    lambda_tier: Optional[str] = None,
    effective_sample_size: Optional[float] = None,
):
    """
    Kelly reduzido para controlar risco.

    `lambda_tier`/`effective_sample_size` são opcionais (Melhoria #6): se
    fornecidos, escalam `fraction` pela confiança do modelo via
    `calculate_adaptive_kelly_fraction` antes de multiplicar pelo Kelly
    completo. Omissos (comportamento por omissão), o resultado é
    exatamente igual ao de antes desta melhoria.
    """

    adaptive_fraction = calculate_adaptive_kelly_fraction(
        fraction, lambda_tier, effective_sample_size
    )
    return kelly_fraction(probability, odd) * adaptive_fraction
