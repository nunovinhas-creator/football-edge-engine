# Historical Dataset Builder

**Âmbito:** módulo novo (`src/historical_dataset/`), independente do
Backtesting Framework já existente (`src/backtest/historical/`). Constrói
um dataset histórico real a partir da BSD API (competições, épocas, jogos
terminados, odds e estatísticas). **Não calcula nem altera nenhuma
fórmula matemática do projeto** — Dixon-Coles, Monte Carlo, Kelly, Edge,
EV, Goal Engine e Machine Learning permanecem exatamente como estavam.
Este módulo só lê e normaliza dados brutos.

---

## Objetivo

Antes desta adição, o projeto não tinha nenhuma integração própria com
uma fonte de jogos históricos com odds e resultados: `src.api`/
`src.collector` servem apenas eventos futuros/ao vivo, e o Backtesting
Framework (`src/backtest/historical/dataset.py`) só aceitava CSV fornecido
manualmente pelo utilizador (ver `docs/04_backtesting_framework.md`).

O Historical Dataset Builder preenche essa lacuna: percorre a BSD API de
ponta a ponta e produz um dataset único, normalizado e exportável,
pronto a ser consumido por qualquer análise posterior — incluindo, mas
não exclusivamente, o Backtesting Framework (ver secção "Integração").

---

## Localização

```
src/historical_dataset/
    rate_limiter.py     # RateLimiter — janela deslizante local, sem depender do limite real da API
    client.py            # BSDHistoricalClient — Authorization: Token, retries (reutiliza src.api.http_retry)
    paginator.py           # iter_endpoint / extract_items — paginação limit/offset genérica
    checkpoint.py            # Checkpoint / NullCheckpoint — checkpoint/resume em disco
    dedup.py                  # Deduplicator / dedupe_records — deduplicação por chave
    normalizer.py               # normalize_event — combina jogo+odds+stats num registo plano
    builder.py                   # HistoricalDatasetBuilder — orquestrador (o ponto de entrada principal)
    storage.py                     # to_csv / to_sqlite / to_parquet / export_all
    backtest_bridge.py               # to_backtest_frame — ponte para o Backtesting Framework existente

build_historical_dataset.py    # CLI de ponta a ponta (equivalente a run_backtest.py para este módulo)

tests/historical_dataset/
    test_rate_limiter.py, test_client.py, test_paginator.py, test_checkpoint.py,
    test_dedup.py, test_normalizer.py, test_builder.py, test_storage.py, test_backtest_bridge.py
```

Coexiste com `src/api/`, `src/collector/` e `src/live/providers/` (que
servem jogos futuros/ao vivo) e com `src/backtest/historical/` (que
avalia apostas já feitas) — nenhum destes módulos foi alterado.

---

## Arquitetura / fluxo de dados

```
BSDHistoricalClient (Authorization: Token, rate limiting, retries)
        │
        ▼
HistoricalDatasetBuilder.build()
        │
        ├─ iter_competitions()      GET /api/v2/leagues/                (paginado)
        │
        └─ para cada liga:
             ├─ iter_seasons(league_id)                GET /api/v2/leagues/{id}/seasons/
             │
             └─ para cada época (season_id):
                  ├─ Checkpoint.is_season_done? ──── sim ──▶ salta (resume)
                  │
                  └─ iter_finished_events(league_id, season_id)
                       GET /api/v2/events/?league_id=&season_id=&status=finished   (paginado)
                       │
                       └─ para cada jogo:
                            ├─ Dedup / Checkpoint.is_event_done? ── sim ──▶ salta
                            │
                            ├─ fetch_odds(event_id)   GET /api/v2/events/{id}/odds/
                            ├─ fetch_stats(event_id)  GET /api/v2/events/{id}/stats/
                            │  (falhas nestes dois pedidos NÃO abortam o pipeline —
                            │   ficam a None e são registadas via `on_error`)
                            │
                            ├─ normalize_event(event, odds, stats, league, season)
                            ├─ Checkpoint.mark_event_done(event_id)
                            └─ yield registo normalizado
                  │
                  └─ Checkpoint.mark_season_done(league_id, season_id)
        │
        ▼
storage.export_all(records, output_dir)  ──▶  .csv + .sqlite + .parquet (se suportado)
        │
        ▼
backtest_bridge.to_backtest_frame(records, market, model_prob=...)  ──▶  src.backtest.historical.dataset.load_historical_dataset(...)
```

`HistoricalDatasetBuilder.build(...)` é um gerador — os registos são
produzidos (e podem ser exportados) em streaming, sem carregar o dataset
inteiro em memória antes de começar a escrever.

---

## Dataset normalizado

Cada linha (`src.historical_dataset.normalizer.NORMALIZED_COLUMNS`)
representa um jogo terminado, com os seguintes campos **sempre que
disponíveis na API** (campos indisponíveis ficam `None`, nunca fazem a
construção do dataset falhar):

| Grupo | Colunas |
|---|---|
| Identificação | `event_id`, `competition_id`, `competition`, `season_id`, `season`, `round_number`, `round_name` |
| Jogo | `date`, `status`, `home_team_id`, `home_team`, `away_team_id`, `away_team`, `venue_id` |
| Resultado | `home_score`, `away_score` (final), `home_score_ht`, `away_score_ht` (intervalo) |
| Odds 1X2 | `odds_home`, `odds_draw`, `odds_away` |
| Odds Over/Under | `odds_over_1_5`, `odds_under_1_5`, `odds_over_2_5`, `odds_under_2_5`, `odds_over_3_5`, `odds_under_3_5` |
| Odds BTTS | `odds_btts_yes`, `odds_btts_no` |
| Bookmaker | `bookmaker` (`"consensus"` quando há odds — ver limitações), `bookmakers_available` (lista de slugs, se `/odds/comparison/` foi pedido) |
| Cartões | `cards_home_yellow`, `cards_home_red`, `cards_away_yellow`, `cards_away_red` |
| Outras estatísticas | `corners_home/away`, `shots_home/away`, `shots_on_target_home/away`, `possession_home/away`, `fouls_home/away`, `offsides_home/away` |
| Estatísticas restantes | `extra_stats_home`, `extra_stats_away` (JSON, por equipa), `extra_match_stats` (JSON, ao nível do jogo — shotmap, momentum, average positions, etc.), `extra_odds` (JSON, estruturas de odds não reconhecidas) |

---

## Paginação, retries, rate limiting, checkpoint/resume e deduplicação

- **Paginação** (`paginator.iter_endpoint`): avança `limit`/`offset`
  sobre `/leagues/` e `/events/` até uma página devolver menos itens do
  que o tamanho pedido; aceita tanto um array JSON simples como
  `{"results": [...]}` (`extract_items`), porque `schema.yaml` documenta
  o primeiro formato mas código de investigação anterior
  (`research/pressure_shots/api.py`) já tinha observado o segundo na
  prática.
- **Retries**: `BSDHistoricalClient` reutiliza
  `src.api.http_retry.get_with_retry` — o mesmo helper de backoff
  exponencial (timeout, connection error, 429/500/502/503/504) já usado
  por todos os outros clientes do projeto. Não foi reimplementado.
- **Rate limiting** (`rate_limiter.RateLimiter`): janela deslizante local
  (por omissão 5 pedidos/segundo), configurável via `build_historical_dataset.py --rate-limit`.
  É uma salvaguarda proativa do lado do cliente, **não** uma
  implementação do limite real da API (`schema.yaml` não documenta
  nenhum limite de pedidos — ver "Limitações").
- **Checkpoint/resume** (`checkpoint.Checkpoint`): persiste em disco (a)
  épocas totalmente concluídas (`completed_seasons.json`, reescrito por
  inteiro a cada época — lista pequena) e (b) jogos já processados
  (`processed_events.log`, ficheiro append-only, uma linha por ID,
  `flush()` imediato). Uma execução interrompida a meio pode ser
  retomada apontando `--checkpoint-dir` para o mesmo diretório: épocas
  já concluídas são saltadas por completo, e jogos já processados dentro
  de uma época interrompida também são saltados individualmente.
- **Deduplicação** (`dedup.Deduplicator`): garante, dentro de uma mesma
  execução, que o mesmo jogo nunca é processado duas vezes (ex.
  sobreposição de páginas), mesmo sem checkpoint ativo.

---

## Exportação (`storage.py`)

`export_all(records, output_dir, base_name="historical_dataset")` escreve
sempre:

- `{base_name}.csv` — via `pandas.DataFrame.to_csv`.
- `{base_name}.sqlite` — via `pandas.DataFrame.to_sql` sobre uma ligação
  `sqlite3` (biblioteca padrão), tabela `historical_matches` (substituída
  se já existir).

E tenta escrever `{base_name}.parquet` (via `pandas.DataFrame.to_parquet`)
**apenas se** um motor Parquet (`pyarrow` ou `fastparquet`) estiver
instalado — devolve `None` para essa chave em vez de falhar quando não
está (requisito "Parquet (se suportado)"). `pyarrow` já é uma dependência
transitiva do `streamlit` (já em `requirements.txt`), pelo que normalmente
está disponível sem instalação adicional.

---

## Uso

```python
from src.historical_dataset import HistoricalDatasetBuilder, export_all
from src.historical_dataset.checkpoint import Checkpoint

builder = HistoricalDatasetBuilder(checkpoint=Checkpoint("data/historical/.checkpoint"))

records = list(builder.build())  # todas as ligas ativas, todas as épocas, todos os jogos terminados

paths = export_all(records, "data/historical")
print(paths)  # {"csv": ..., "sqlite": ..., "parquet": ... ou None}
```

### CLI (`build_historical_dataset.py`)

```bash
# Todas as ligas ativas, execução única (sem checkpoint)
python build_historical_dataset.py --output-dir data/historical

# Apenas duas ligas específicas, com checkpoint/resume
python build_historical_dataset.py --leagues 39,140 --checkpoint-dir data/historical/.checkpoint

# Execução parcial/rápida (ex. para validar a configuração antes de correr por completo)
python build_historical_dataset.py --max-events 200

# Também obter a comparação de bookmakers por jogo (preenche `bookmakers_available`)
python build_historical_dataset.py --odds-comparison
```

Ver `python build_historical_dataset.py --help` para todas as opções
(`--rate-limit`, `--page-size`, `--no-odds`, `--no-stats`, `--country`,
`--include-inactive`).

---

## Integração com o Backtesting Framework (`backtest_bridge.py`)

O Backtesting Framework já existente
(`src/backtest/historical/dataset.load_historical_dataset`) espera uma
linha **por aposta** (jogo + mercado + odd + probabilidade do modelo),
não uma linha por jogo. `to_backtest_frame(records, market, model_prob=...)`
faz essa conversão:

```python
from src.historical_dataset.backtest_bridge import to_backtest_frame
from src.backtest.historical.dataset import load_historical_dataset
from src.backtest.historical import BacktestEngine, FlatStake

# 1. Construir/carregar o dataset histórico (ver secção "Uso")
records = list(builder.build())

# 2. Escolher um mercado e fornecer model_prob — este builder NUNCA calcula
#    probabilidade; `model_prob` tem de vir do motor de previsão já
#    existente (src.engine.*) aplicado a estes jogos, de uma coluna já
#    presente nos registos, ou de uma função row -> float.
bridge_df = to_backtest_frame(records, market="HOME", model_prob=minha_probabilidade_do_modelo)

# 3. A partir daqui, é o Backtesting Framework existente, sem alterações:
dados = load_historical_dataset(bridge_df)
report = BacktestEngine(staking=FlatStake(unit=1.0)).run(dados)
report.print_summary()
```

`model_prob` aceita: um escalar (mesma probabilidade para todas as
linhas), o nome de uma coluna já presente em `records`, uma
`Series`/lista alinhada por posição, ou uma função `row -> float`.
`engine_decision` e `result`, se omitidos, continuam a ser preenchidos
automaticamente por `load_historical_dataset` (a partir de
`home_goals`/`away_goals` e da odd/probabilidade fornecidas), exatamente
como já acontecia — este módulo não duplica essa lógica.

Linhas sem odd publicada para o mercado escolhido (ou sem resultado
final conhecido) são descartadas por `to_backtest_frame` antes de chegar
a `load_historical_dataset`.

---

## Limitações

1. **Forma exata de `/events/{id}/odds/`, `/events/{id}/odds/comparison/`
   e `/events/{id}/stats/` não documentada.** `schema.yaml` (a
   especificação OpenAPI oficial incluída no repositório) marca estes
   três endpoints como `"No response body"` — só a descrição textual está
   documentada (ex. "Consensus decimal odds. Markets: 1X2, Over/Under
   {1.5,2.5,3.5}, BTTS."; "Outcome keys are codes (HOME/DRAW/AWAY/
   over/under/...), not team names ... Bookmaker keys are slugs."). O
   `normalizer.py` assume, com base nessas descrições, uma forma
   plausível e faz extração defensiva por múltiplos aliases; qualquer
   campo fora do esperado é preservado (não descartado) nas colunas
   `extra_*`. **Recomendação:** ao ligar a uma chave de API real pela
   primeira vez, validar um punhado de respostas reais destes três
   endpoints e ajustar `normalizer.py` se a forma real divergir do
   assumido.
2. **Coluna `bookmaker` é `"consensus"`, não um bookmaker específico**,
   porque `/events/{id}/odds/` devolve odds de consenso (agregadas), não
   por casa de apostas. Para saber quais bookmakers estão por trás desse
   consenso, ativar `--odds-comparison` (`include_odds_comparison=True`),
   que preenche `bookmakers_available` a partir de
   `/events/{id}/odds/comparison/` — mas isto duplica os pedidos HTTP por
   jogo (mais um pedido cada), pelo que fica desligado por omissão.
3. **Limite de pedidos por minuto/segundo da BSD API não documentado.**
   `schema.yaml` não define nenhum `X-RateLimit-*` nem limite explícito.
   O `RateLimiter` deste módulo (por omissão 5 pedidos/segundo) é uma
   salvaguarda proativa e configurável do lado do cliente, não uma cópia
   do limite real — se a API responder 429 mesmo assim, `get_with_retry`
   já trata isso como falha transitória com backoff exponencial.
4. **Checkpoint ao nível de época, não ao nível de página.** Se o
   processo for interrompido a meio de uma época muito longa, o resume
   salta jogos já marcados individualmente como concluídos
   (`processed_events.log`), mas repete a paginação de `/events/` para
   essa época a partir do início (custo: alguns pedidos de listagem
   redundantes, não pedidos de odds/stats já feitos).
5. **`extra_match_stats` pode incluir estruturas grandes** (shotmap,
   momentum, average positions, per-minute xG) como JSON serializado numa
   única célula — adequado para CSV/SQLite/Parquet, mas não pensado para
   ser lido/filtrado eficientemente a partir daí; para análise dessas
   estruturas, considerar tratamento dedicado fora deste builder.
6. **Não há normalização de fuso horário além do que a API já devolve**
   (`event_date` é ISO 8601 UTC, conforme `EventDetailV2Schema`); nenhuma
   conversão adicional é feita.
7. **Este módulo não calcula probabilidade de modelo.** Como exigido, não
   invoca Dixon-Coles, Monte Carlo, ML, etc. — `backtest_bridge.py` exige
   explicitamente que `model_prob` seja fornecido por quem chama.

---

## Testes

`tests/historical_dataset/` (78 testes, todos com dados/clientes falsos —
nenhum pedido de rede real):

- `test_rate_limiter.py` — janela deslizante, espera calculada, reset.
- `test_client.py` — cabeçalho `Authorization: Token`, rate limiting
  aplicado antes do pedido, erro 4xx, corpo vazio.
- `test_paginator.py` — array simples vs. `{"results": [...]}`, avanço de
  offset, paragem no fim, `max_pages`.
- `test_checkpoint.py` — persistência e resume de épocas/jogos entre
  instâncias, `NullCheckpoint` no-op.
- `test_dedup.py` — deduplicação por chave, chave por omissão vs.
  personalizada.
- `test_normalizer.py` — mapeamento de `EventDetailV2Schema`, várias
  formas plausíveis de odds/stats, preservação de campos desconhecidos em
  `extra_*`, ausência de odds/stats não falha.
- `test_builder.py` — percurso completo com cliente falso em memória,
  deduplicação, checkpoint/resume (incluindo retomar a meio de uma
  época), falhas parciais de odds/stats não abortam o pipeline,
  `max_events`.
- `test_storage.py` — CSV, SQLite (consulta real via `sqlite3`), Parquet
  (`skipTest` automático se nenhum motor estiver instalado), dataset
  vazio.
- `test_backtest_bridge.py` — seleção de coluna de odds por mercado, as
  quatro formas de `model_prob`, remoção de linhas sem odd/resultado, e
  um teste de integração de ponta a ponta com
  `load_historical_dataset` real (sem mocks).

Correr apenas os testes deste módulo:

```
python -m unittest discover -s tests/historical_dataset -v
```

Nenhum teste pré-existente foi alterado; a suite completa
(`python -m unittest discover -s tests -v`) continua a passar na íntegra
após esta adição.
