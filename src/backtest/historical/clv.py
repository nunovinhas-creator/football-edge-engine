"""
Closing Line Value (CLV) do Backtesting Framework / Evaluation Framework.

CLV mede a diferença entre a odd disponível no momento em que o motor fez
a previsão ("opening odd", já capturada em `HistoricalBet.odd` /
`EvaluatedBet.odd`) e a última odd disponível antes do início do jogo
("closing odd", `HistoricalBet.closing_odd` / `EvaluatedBet.closing_odd`,
sempre opcional).

É uma métrica de EFICIÊNCIA DE EXECUÇÃO da aposta (a odd conseguida vs. o
mercado no fecho) — não prevê nada, não recalcula nem substitui nenhuma
fórmula do motor de previsão (Dixon-Coles, Monte Carlo, Goal Engine,
Machine Learning, Kelly, Edge, EV). Ambas as odds usadas aqui já existem
nos dados de entrada (fornecidas pelo utilizador, tipicamente obtidas da
BSD API já existente através de `src.collector.odds.OddsCollector` /
`src.live.providers.api_odds_provider.APIOddsProvider` /
`src.historical_dataset.client.BSDHistoricalClient` — nenhuma API externa
nova é usada nem introduzida por este módulo).

Ver `docs/09_clv.md` para a definição completa, interpretação e
limitações.
"""

from typing import Optional

CLV_POSITIVE = "POSITIVE"
CLV_NEGATIVE = "NEGATIVE"
CLV_NEUTRAL = "NEUTRAL"


def calculate_clv_absolute(opening_odd: float, closing_odd: Optional[float]) -> Optional[float]:
    """
    CLV absoluto = opening_odd - closing_odd (mesma unidade das odds decimais).

    Positivo: a odd conseguida na previsão era mais alta (melhor para o
    apostador) do que a odd de fecho — a linha "encurtou" a favor da
    seleção apostada depois de a aposta ter sido decidida (o motor "bateu"
    o fecho).
    Negativo: a odd de fecho subiu acima da odd conseguida — o mercado
    moveu-se contra a seleção apostada depois da previsão.
    Zero: odd de fecho igual à odd da previsão (CLV neutro).
    `None`: sem odd de fecho disponível — CLV não pode ser calculado.
    """
    if closing_odd is None:
        return None
    return round(float(opening_odd) - float(closing_odd), 6)


def calculate_clv_percentage(opening_odd: float, closing_odd: Optional[float]) -> Optional[float]:
    """
    CLV percentual = (opening_odd - closing_odd) / closing_odd * 100.

    Expressa o CLV absoluto como percentagem da odd de fecho — a forma
    convencional de comparar CLV entre apostas com odds de magnitudes
    muito diferentes (ex. odd 1.50 vs. odd 5.00). `None` quando não há
    odd de fecho (ou esta não é positiva, o que nunca deveria acontecer
    com uma odd decimal válida).
    """
    if closing_odd is None or closing_odd <= 0:
        return None
    return round(100.0 * (float(opening_odd) - float(closing_odd)) / float(closing_odd), 4)


def classify_clv(clv_absolute: Optional[float]) -> Optional[str]:
    """
    Segmenta o CLV absoluto em `"POSITIVE"` / `"NEGATIVE"` / `"NEUTRAL"`.
    Devolve `None` quando o CLV não é calculável (sem odd de fecho).
    """
    if clv_absolute is None:
        return None
    if clv_absolute > 0:
        return CLV_POSITIVE
    if clv_absolute < 0:
        return CLV_NEGATIVE
    return CLV_NEUTRAL


def beat_closing_market(clv_absolute: Optional[float]) -> Optional[bool]:
    """
    `True` se a odd conseguida não foi pior do que a odd de fecho
    (CLV >= 0) — ou seja, a aposta "bateu ou empatou" o mercado no fecho.

    Distingue-se de `classify_clv(...) == "POSITIVE"` por incluir o caso
    neutro (CLV == 0) como sucesso: é uma medida mais permissiva de "não
    perder valor face ao mercado", usada pela métrica
    `metrics.beat_market_pct` (ver `docs/09_clv.md`, secção
    "CLV positivo vs. bater o mercado").
    """
    if clv_absolute is None:
        return None
    return clv_absolute >= 0
