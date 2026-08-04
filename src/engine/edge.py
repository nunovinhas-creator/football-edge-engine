"""
Módulo de Cálculo de Edge e Expected Value (EV) do Engine.

Definição oficial de Edge (ver docs/AUDIT_MATEMATICA.md, secção
"Definição Oficial de Edge"):

    edge = prob_model - implied_probability(odd_house)

isto é, a diferença entre a probabilidade estimada pelo modelo e a
probabilidade implícita do mercado (1 / odd). Esta é uma grandeza
diferente do Expected Value (EV):

    ev = (prob_model * odd_house) - 1.0

`calculate_edge` é a ÚNICA implementação oficial de Edge do projeto —
todos os módulos que precisem de Edge devem importar e reutilizar esta
função, em vez de recalcular a fórmula localmente.

Ambas as funções recebem `prob_model` como fração (0.0 - 1.0] e devolvem
o resultado na mesma escala; para apresentação em pontos percentuais,
multiplicar o resultado por 100.

Remoção do overround (Melhoria #7 da auditoria matemática, ver
docs/AUDIT_MATEMATICA.md secção "Remoção do overround..."):

`implied_probability(odd_house)` é a probabilidade implícita de UMA odd
isolada — inclui a margem da casa (overround/vig), porque a soma das
probabilidades implícitas de todas as opções de um mercado (ex. as 3
odds de um 1X2) é sempre > 1.0. `remove_overround()` recebe as odds de
TODAS as opções de um mesmo mercado e devolve as probabilidades "fair"
(sem margem, a somar 1.0). `calculate_edge()` passa a aceitar,
opcionalmente, essas odds completas do mercado (`market_odds`) para usar
a probabilidade fair em vez da implícita — sem alterar a assinatura
mínima nem o comportamento para quem continua a chamar apenas com
`(prob_model, odd_house)`.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Union


def implied_probability(odd: float) -> float:
    """
    Converte odd decimal em probabilidade implícita (0.0 - 1.0).
    """
    if odd <= 1.0:
        return 0.0

    return round(1.0 / odd, 4)


def _devig_sequence(odds: Sequence[float]) -> Optional[List[Optional[float]]]:
    """
    Núcleo do cálculo de remoção de overround, comum às variantes de
    `remove_overround()` (lista ou dict). Método usado: normalização
    proporcional das probabilidades implícitas (o método básico e mais
    comum de remoção de overround — "multiplicative"/"basic" method):

        overround = Σ implied_probability(odd_i)
        fair_probability(odd_i) = implied_probability(odd_i) / overround

    Devolve `None` quando há menos de 2 odds válidas (odd > 1.0) no
    conjunto — não há mercado suficiente para calcular um overround com
    significado, e o chamador deve manter o comportamento atual (usar a
    probabilidade implícita simples, com margem).

    Preserva a posição/comprimento da sequência recebida: entradas
    inválidas (odd <= 1.0, None, não numéricas) ficam a `None` no
    resultado, em vez de serem descartadas — para que o índice de cada
    odd continue a corresponder ao índice da sua probabilidade fair.
    """

    valid_odds = [
        o for o in odds if isinstance(o, (int, float)) and not isinstance(o, bool) and o > 1.0
    ]

    if len(valid_odds) < 2:
        return None

    overround = sum(1.0 / o for o in valid_odds)

    if overround <= 0:
        return None

    result: List[Optional[float]] = []
    for o in odds:
        if isinstance(o, (int, float)) and not isinstance(o, bool) and o > 1.0:
            result.append(round((1.0 / o) / overround, 6))
        else:
            result.append(None)

    return result


def remove_overround(
    odds: Union[Sequence[float], Dict[str, float]]
) -> Optional[Union[List[Optional[float]], Dict[str, Optional[float]]]]:
    """
    Função única e reutilizável de remoção do overround (margem da casa).

    Recebe as odds decimais de TODAS as opções de um mesmo mercado —
    por exemplo:
        - 1X2:        {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        - Over/Under: {"OVER_2.5": 1.90, "UNDER_2.5": 1.95}
        - BTTS:       {"YES": 1.85, "NO": 1.95}
    (aceita também uma lista/tupla simples de odds, na mesma ordem.)

    Devolve as probabilidades "fair" (sem margem), na mesma estrutura
    (dict ou lista) recebida, normalizadas para somar 1.0 entre as odds
    válidas.

    Devolve `None` quando não existem odds suficientes (menos de 2 odds
    válidas, i.e. > 1.0) para calcular um overround com significado — é
    responsabilidade do chamador manter o comportamento anterior
    (probabilidade implícita simples, com margem) nesse caso.
    """

    if isinstance(odds, dict):
        keys = list(odds.keys())
        fair_values = _devig_sequence([odds[k] for k in keys])
        if fair_values is None:
            return None
        return dict(zip(keys, fair_values))

    return _devig_sequence(list(odds))


def _fair_probability(
    odd_house: float, market_odds: Optional[Union[Iterable[float], Dict[str, float]]]
) -> Optional[float]:
    """
    Devolve a probabilidade fair (sem overround) correspondente a
    `odd_house`, calculada a partir do conjunto completo de odds do
    mercado (`market_odds`), reutilizando `remove_overround()`.

    Devolve `None` (sinal para o chamador usar `implied_probability`
    simples, comportamento atual) quando `market_odds` não é fornecido,
    não tem odds válidas suficientes, ou não contém `odd_house`.
    """

    if not market_odds:
        return None

    odds_sequence = list(market_odds.values()) if isinstance(market_odds, dict) else list(market_odds)

    fair_probs = _devig_sequence(odds_sequence)
    if fair_probs is None:
        return None

    for odd, fair_prob in zip(odds_sequence, fair_probs):
        if fair_prob is not None and odd == odd_house:
            return fair_prob

    return None


def calculate_edge(
    prob_model: float,
    odd_house: float,
    market_odds: Optional[Union[Iterable[float], Dict[str, float]]] = None,
) -> float:
    """
    Calcula o Edge oficial: a diferença entre a probabilidade do modelo
    e a probabilidade "fair" (sem overround) de mercado, quando esta
    puder ser calculada — ou a probabilidade implícita simples (com
    margem), caso contrário.

        edge = prob_model - fair_probability(odd_house, market_odds)

    prob_model:
        Probabilidade do modelo, em fração (0.0 - 1.0].

    odd_house:
        Odd decimal da casa (> 1.0) para a opção que está a ser avaliada.

    market_odds (opcional, Melhoria #7 da auditoria matemática):
        Todas as odds do mesmo mercado (dict outcome->odd, ou lista),
        incluindo `odd_house`. Quando fornecido e houver pelo menos 2
        odds válidas nesse conjunto (incluindo `odd_house`), o overround
        é removido (`remove_overround()`) e a probabilidade fair
        resultante é usada em vez da probabilidade implícita simples.

        Quando omitido, ou quando não há odds suficientes para remover a
        margem (menos de 2 odds válidas, ou `odd_house` ausente do
        conjunto), o comportamento é EXATAMENTE o mesmo que antes desta
        melhoria: `edge = prob_model - implied_probability(odd_house)`.

    Lança ValueError se `prob_model` ou `odd_house` forem inválidos, em
    vez de devolver um valor sentinela silencioso — foi exatamente essa
    falha silenciosa (chamar esta função com uma probabilidade no lugar
    de uma odd) que causou o bug histórico em `src/engine/market.py`.
    """

    if not (0.0 < prob_model <= 1.0):
        raise ValueError(
            f"prob_model inválido: {prob_model!r}. "
            "Deve ser uma probabilidade em fração, no intervalo (0.0, 1.0]."
        )

    if odd_house <= 1.0:
        raise ValueError(
            f"odd_house inválida: {odd_house!r}. "
            "Deve ser uma odd decimal > 1.0 (não uma probabilidade)."
        )

    market_probability = _fair_probability(odd_house, market_odds)
    if market_probability is None:
        market_probability = implied_probability(odd_house)

    return round(prob_model - market_probability, 4)


def edge(
    prob_model: float,
    odd_house: float,
    market_odds: Optional[Union[Iterable[float], Dict[str, float]]] = None,
) -> float:
    """
    Wrapper para compatibilidade. Ver calculate_edge().
    """
    return calculate_edge(prob_model, odd_house, market_odds)


def calculate_ev(
    prob_model: float,
    odd_house: float,
    market_odds: Optional[Union[Iterable[float], Dict[str, float]]] = None,
) -> float:
    """
    Calcula o Expected Value (EV) por unidade apostada.

        ev = (prob_model * odd_house) - 1.0

    prob_model:
        probabilidade do modelo em fração (0.0 - 1.0)

    odd_house:
        odd decimal da casa

    market_odds (opcional, Melhoria #7 da auditoria matemática):
        aceite apenas por simetria de interface com `calculate_edge()` —
        para que um chamador que já tenha o conjunto completo de odds de
        um mercado (ex. `src/engine/market.py::analyze_market`) possa
        passá-lo a ambas as funções sem tratamento especial. NÃO altera
        o valor devolvido: a fórmula de EV usa apenas a probabilidade do
        próprio modelo e a odd real paga pela casa — não depende, direta
        ou indiretamente, da probabilidade implícita/fair de mercado, pelo
        que remover o overround não tem qualquer efeito matemático sobre
        o EV (ver docs/AUDIT_MATEMATICA.md §6.1/§11: fórmula já
        classificada como correta e não afetada pelo overround).
    """
    if odd_house <= 1.0 or prob_model <= 0:
        return -1.0

    return round((prob_model * odd_house) - 1.0, 4)


def expected_value(
    prob_model: float,
    odd_house: float,
    market_odds: Optional[Union[Iterable[float], Dict[str, float]]] = None,
) -> float:
    """
    Alias de EV para compatibilidade com versões anteriores.
    """
    return calculate_ev(prob_model, odd_house, market_odds)
