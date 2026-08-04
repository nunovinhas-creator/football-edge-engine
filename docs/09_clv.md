# Closing Line Value (CLV)

**Âmbito:** funcionalidade nova do Backtesting Framework / Evaluation
Framework (`src/backtest/historical/`, `src/evaluation/`). **Não altera
nenhum algoritmo de previsão** — Dixon-Coles, Monte Carlo, Goal Engine,
Machine Learning, Kelly, Edge, Expected Value, Dashboard, Live Pipeline e
o Lambda Estimator permanecem exatamente como estavam. CLV é uma métrica
de **medição a posteriori**, calculada exclusivamente a partir de duas
odds já capturadas por aposta — nunca influencia `probability`, `edge`,
`ev`, `kelly`, `stake` nem `engine_decision`.

---

## 1. Definição

> **CLV (Closing Line Value)** = diferença entre a odd disponível quando
> o motor fez a previsão ("opening odd") e a última odd disponível antes
> do início do jogo ("closing odd").

No código:

- **Opening odd** = `HistoricalBet.odd` / `EvaluatedBet.odd` (já existia
  — é a odd que o motor usou para calcular probabilidade, Edge, EV e
  Kelly). Exposta também como `HistoricalBet.opening_odd` (propriedade,
  alias de `odd` — não duplica armazenamento).
- **Closing odd** = `HistoricalBet.closing_odd` / `EvaluatedBet.closing_odd`
  (novo campo, **opcional**) — a última odd conhecida para o mesmo
  jogo/mercado/seleção antes do apito inicial.

Fórmulas (`src/backtest/historical/clv.py`):

```
clv_absolute    = opening_odd - closing_odd
clv_percentage  = (opening_odd - closing_odd) / closing_odd * 100
```

- **CLV positivo** (`clv_absolute > 0`): a odd conseguida na previsão era
  mais alta (melhor para o apostador) do que a odd de fecho — a linha
  "encurtou" a favor da seleção apostada depois da previsão ter sido
  feita. O motor "bateu" o fecho.
- **CLV negativo** (`clv_absolute < 0`): a odd de fecho subiu acima da
  odd conseguida — o mercado moveu-se contra a seleção apostada depois da
  previsão.
- **CLV neutro** (`clv_absolute == 0`): a odd não se moveu.
- **CLV indisponível** (`None`): sem `closing_odd` — nenhum dos campos de
  CLV é calculado, e nenhuma outra métrica é afetada.

---

## 2. Interpretação

CLV é a métrica mais usada na literatura de apostas desportivas
profissionais para avaliar a qualidade da EXECUÇÃO de uma aposta,
independentemente do resultado do jogo: como a odd de fecho é o melhor
estimador disponível da probabilidade real de um evento (o mercado
incorpora toda a informação disponível até ao apito inicial), bater
consistentemente o fecho é um sinal mais fiável de vantagem real do que a
taxa de acerto ou o lucro de uma amostra pequena — que estão sujeitos a
variância de curto prazo mesmo com um modelo genuinamente bom (ou mau).

Duas apostas *ganhas* podem ter CLV muito diferente (uma odd tomada a
2.10 que fecha a 1.90 teve CLV positivo mesmo que perca; uma odd tomada a
2.10 que fecha a 2.40 teve CLV negativo mesmo que ganhe) — por isso CLV é
sempre reportado **em paralelo** com ROI/Yield/Hit Rate, nunca como
substituto.

### CLV positivo vs. "bater o mercado"

O Evaluation Framework reporta duas percentagens relacionadas mas
distintas (`src/backtest/historical/metrics.py`):

| Métrica | Critério | Interpretação |
|---|---|---|
| `clv_positive_pct` | `clv_absolute > 0` (estrito) | % de apostas em que se **ganhou** valor face ao fecho |
| `beat_market_pct` | `clv_absolute >= 0` (inclui o neutro) | % de apostas em que **não se perdeu** valor face ao fecho |

Ambas as percentagens (e as restantes métricas de CLV) são calculadas
apenas sobre o subconjunto de apostas com `closing_odd` disponível —
`clv_coverage_pct` reporta que fração do total isso representa, para que
uma cobertura baixa não seja confundida com um sinal de CLV bom/mau.

---

## 3. Mecanismo de armazenamento

### `HistoricalBet` (entrada)

```python
from src.backtest.historical.models import HistoricalBet

bet = HistoricalBet(
    match="Benfica vs Porto", date="2026-03-01", market="HOME",
    odd=2.10,                 # opening odd (obrigatória, já existia)
    model_prob=0.55, engine_decision="BET", result="WIN",
    closing_odd=1.90,         # novo, opcional
    bookmaker="consensus",    # novo, opcional
)
```

`HistoricalBet.from_dict(...)` aceita as mesmas variantes em
português/inglês já usadas pelo resto do módulo:

- `closing_odd`: `closing_odd`, `odd_fecho`, `odd_final`,
  `odd_fechamento`, `closing_line`.
- `bookmaker`: `bookmaker`, `casa_apostas`, `casa_de_apostas`, `bookie`.

Ambos os campos são **totalmente opcionais**: registos/CSV/DataFrames
sem estas colunas continuam a ser lidos exatamente como antes desta
funcionalidade (ver secção "Retrocompatibilidade").

### `EvaluatedBet` (saída de `evaluator.evaluate_bet`)

`evaluate_bet(bet)` calcula automaticamente os campos de CLV sempre que
`bet.closing_odd` está presente:

```python
EvaluatedBet(
    ...,
    closing_odd=bet.closing_odd,
    bookmaker=bet.bookmaker,
    clv_absolute=...,        # None se closing_odd ausente
    clv_percentage=...,      # None se closing_odd ausente
    clv_classification=...,  # "POSITIVE" | "NEGATIVE" | "NEUTRAL" | None
)
```

`EvaluatedBet.to_dict()` inclui estes campos (mais `opening_odd`, alias
de `odd`, para clareza nos relatórios) — refletidos em todos os
DataFrames/exportações produzidos pelo `BacktestReport`/`EvaluationReport`.

### Origem da closing odd

Este módulo **não faz pedidos à BSD API** — recebe `closing_odd` já
preenchida por quem constrói o `HistoricalBet`. Para a obter, reutilize a
integração BSD API já existente no repositório (nenhuma API nova é
introduzida), consultando as odds do mesmo jogo/mercado outra vez perto
do kickoff:

- `src.collector.odds.OddsCollector.get_event_odds(event_id)`
- `src.live.providers.api_odds_provider.APIOddsProvider.get_live_odds(event_id)`
- `src.historical_dataset.client.BSDHistoricalClient` (Historical Dataset
  Builder, jogos já terminados)

O agendamento de *quando* voltar a consultar a API perto do kickoff é uma
decisão do chamador (ex. um `cron`/scheduler externo ao motor) — está
fora do âmbito desta funcionalidade, que se limita ao armazenamento e
cálculo, e não altera nenhum componente do Live Pipeline.

---

## 4. Métricas do Evaluation Framework

`summary_metrics(df)` (`src/backtest/historical/metrics.py`) já inclui,
para qualquer subconjunto de apostas avaliadas (globais, colocadas, ou
qualquer segmento — ver abaixo), as seguintes chaves:

| Chave | Descrição |
|---|---|
| `clv_coverage_pct` | % de apostas do subconjunto com `closing_odd` disponível |
| `avg_clv_absolute` / `median_clv_absolute` | CLV absoluto médio/mediano (só sobre apostas com CLV calculável) |
| `avg_clv_percentage` / `median_clv_percentage` | CLV percentual médio/mediano |
| `clv_positive_pct` | % com CLV estritamente positivo |
| `clv_negative_pct` | % com CLV negativo |
| `clv_neutral_pct` | % com CLV exatamente zero |
| `beat_market_pct` | % com CLV >= 0 ("bateu ou empatou" o fecho) |

Como os segmentos (`segments.py`) aplicam `summary_metrics` a cada grupo,
**CLV por competição**, **CLV por mercado** e **CLV por bookmaker** ficam
disponíveis automaticamente nos segmentos já existentes `by_competition`
/ `by_market`, mais o novo `by_bookmaker` — sem nenhuma métrica
duplicada.

Sem `closing_odd` em nenhuma aposta do subconjunto, todas as chaves acima
devolvem `None` (exceto `clv_coverage_pct`, que devolve `0.0`) — nunca um
erro.

---

## 5. Segmentação

Além dos segmentos já existentes (que agora incluem CLV nas suas
métricas), há um novo segmento dedicado:

- **`by_clv_classification`**: agrupa as apostas em `POSITIVE` /
  `NEGATIVE` / `NEUTRAL` (`clv_classification`). Apostas sem
  `closing_odd` (classificação `None`) ficam de fora deste segmento em
  particular — não porque sejam ignoradas no resto do relatório, mas
  porque não pertencem a nenhum dos três grupos.
- **`by_bookmaker`**: agrupa por `bookmaker`.

Ambos aparecem em `BacktestReport.segments` /
`EvaluationReport.all_segment_tables()` exatamente como qualquer outro
segmento, e por isso em todas as exportações (ver secção seguinte).

---

## 6. Relatórios

Como CLV foi integrado nos módulos genéricos de métricas/segmentos (não
como um relatório à parte), **todas** as exportações já existentes
passam a incluir CLV automaticamente, sem nenhum exportador dedicado:

- **CSV** (`BacktestReport.to_csv` / `EvaluationReport.to_csv`):
  `bets.csv` ganha as colunas `opening_odd`, `closing_odd`, `bookmaker`,
  `clv_absolute`, `clv_percentage`, `clv_classification`; `summary.csv`
  ganha as chaves de CLV listadas acima; `segment_by_bookmaker.csv` e
  `segment_by_clv_classification.csv` são novos ficheiros.
- **Excel** (`.to_excel`): mesmas colunas nas folhas `bets`/`summary`,
  mais as folhas `by_bookmaker` e `by_clv_classification`.
- **Markdown** (`EvaluationReport.to_markdown`): a secção "Segmentos" já
  itera sobre todos os segmentos disponíveis — `by_bookmaker` e
  `by_clv_classification` aparecem automaticamente quando há dados.
- **HTML** (`.to_html`): idem — a tabela de segmentos e a tabela "Apostas
  (todas)" já renderizam qualquer coluna/segmento novo sem alterações ao
  gerador de HTML.

---

## 7. Limitações

1. **Sem `closing_odd`, não há CLV** — este módulo não estima nem infere
   a odd de fecho a partir de nenhuma outra informação (não seria uma
   medição, seria uma previsão). Datasets antigos, ou apostas em que o
   fecho nunca foi capturado, simplesmente não contribuem para as
   métricas de CLV (ver `clv_coverage_pct`).
2. **Não há fetching automático da BSD API perto do kickoff.** O
   mecanismo fornecido é de armazenamento e cálculo; a captura da
   `closing_odd` no momento certo é responsabilidade de quem constrói o
   dataset histórico (ver secção 3), reutilizando a integração BSD API já
   existente.
3. **Um único par de odds por aposta.** O modelo assume uma odd de
   abertura e uma odd de fecho por aposta/seleção — não uma série
   temporal de odds intermédias. Analisar a trajetória completa da linha
   está fora do âmbito desta funcionalidade.
4. **CLV mede execução, não acerto.** Uma aposta com CLV muito positivo
   pode perder, e uma com CLV negativo pode ganhar (ver secção 2) — CLV
   deve ser lido em conjunto com ROI/Yield/Hit Rate/Brier Score, nunca
   isoladamente.
5. **Percentagens de CLV ignoram o tamanho do stake.** `clv_positive_pct`
   e `beat_market_pct` contam apostas, não capital — uma aposta grande
   com CLV negativo pesa o mesmo que uma pequena com CLV positivo nestas
   percentagens (tal como `hit_rate_pct`/`yield_pct` já fazem hoje para
   Hit Rate/Yield, por convenção deste framework).

---

## 8. Exemplos

### Exemplo 1 — CLV positivo

```python
from src.backtest.historical.models import HistoricalBet
from src.backtest.historical.evaluator import evaluate_bet

bet = HistoricalBet(
    match="Benfica vs Porto", date="2026-03-01", market="HOME",
    odd=2.10, closing_odd=1.90,
    model_prob=0.55, engine_decision="BET", result="WIN",
)
result = evaluate_bet(bet)
result.clv_absolute        # 0.20  (2.10 - 1.90)
result.clv_percentage      # 10.53 ((2.10 - 1.90) / 1.90 * 100)
result.clv_classification  # "POSITIVE"
```

### Exemplo 2 — CLV negativo

```python
bet = HistoricalBet(
    match="Sporting vs Braga", date="2026-03-02", market="AWAY",
    odd=3.00, closing_odd=3.40,
    model_prob=0.30, engine_decision="BET", result="LOSS",
)
result = evaluate_bet(bet)
result.clv_absolute        # -0.40
result.clv_classification  # "NEGATIVE"
```

### Exemplo 3 — sem odd de fecho (retrocompatível)

```python
bet = HistoricalBet(
    match="Arsenal vs Chelsea", date="2026-03-03", market="HOME",
    odd=1.80, model_prob=0.60, engine_decision="BET", result="WIN",
)
result = evaluate_bet(bet)
result.closing_odd          # None
result.clv_absolute         # None
result.clv_classification   # None
# probability/edge/ev/kelly/stake/profit calculados normalmente.
```

### Exemplo 4 — métricas agregadas

```python
from src.backtest.historical.engine import BacktestEngine

report = BacktestEngine().run([...])  # lista de HistoricalBet/dicts
report.global_metrics["clv_coverage_pct"]
report.global_metrics["avg_clv_percentage"]
report.global_metrics["beat_market_pct"]
report.segments["by_clv_classification"]   # DataFrame POSITIVE/NEGATIVE/NEUTRAL
report.segments["by_bookmaker"]            # DataFrame por bookmaker
```
