# Workflow Manual de Construção do Dataset Histórico (BSD API)

**Âmbito:** infraestrutura de execução (GitHub Actions + CLI + relatório de
qualidade) em torno do Historical Dataset Builder já existente
(`src/historical_dataset/`, ver `docs/07_historical_dataset_builder.md`).
**Este documento não descreve nenhuma recolha real** — nenhuma execução da
BSD API foi feita ao preparar esta infraestrutura; a recolha é sempre
iniciada manualmente, depois, pelo utilizador (`workflow_dispatch` ou CLI
local). Não altera nenhum algoritmo do motor (Dixon-Coles, Monte Carlo, λ
estimator, Kelly, Edge, EV, Goal Engine, Machine Learning, Decision
Engine, Backtesting Engine, Evaluation Framework).

---

## 1. Arquitetura

```
.github/workflows/build_historical_dataset.yml   # workflow_dispatch apenas — sem schedule, sem push
        │
        │  inputs: competition_id, season_id, output_format, page_size, resume
        ▼
build_historical_dataset.py                       # ponto de entrada CLI (wrapper fino)
        │
        ▼
src/historical_dataset/cli.py                      # build_parser() / main(argv) — testável, sem I/O de rede nos testes
        │
        ├─ BSDHistoricalClient (client.py)            # Authorization: Token, rate limiting, conta pedidos (request_count)
        ├─ Checkpoint / NullCheckpoint (checkpoint.py)   # --resume true/false
        ├─ ProgressLogger (cli.py)                         # progresso: competição/época/página/jogos/odds/ETA
        │
        ▼
HistoricalDatasetBuilder.build(...)                  # já existente (builder.py) — apenas com dois aditivos:
        │                                                #   - season_ids: filtra por --season-id
        │                                                #   - progress_callback: emite eventos de progresso
        ▼
storage.to_csv / to_sqlite / to_parquet               # conforme --output (csv/sqlite/parquet/all)
        │
        ▼
src/historical_dataset/report.py                        # build_dataset_report() -> dataset_report.json
        │
        ▼
actions/upload-artifact (apenas ficheiros existentes, sem commit)
```

### O que já existia e não foi tocado

- `HistoricalDatasetBuilder` (percurso competições → épocas → jogos → odds
  → estatísticas), `normalize_event`, `Checkpoint`, `Deduplicator`,
  `RateLimiter`, `storage.to_csv/to_sqlite/to_parquet/export_all` — toda a
  lógica de extração/normalização é exatamente a mesma.

### O que foi adicionado nesta tarefa (apenas infraestrutura/observabilidade)

| Ficheiro | Adição |
|---|---|
| `src/historical_dataset/client.py` | `request_count` (contador de pedidos HTTP feitos, para `dataset_report.json`) |
| `src/historical_dataset/dedup.py` | `Deduplicator.duplicate_count()` (contador de duplicados detetados) |
| `src/historical_dataset/paginator.py` | `iter_endpoint(..., page_callback=...)` opcional (progresso por página) |
| `src/historical_dataset/builder.py` | `build(..., season_ids=..., )` (filtro por época) + `progress_callback` no construtor (eventos `competition_start`/`season_start`/`page`/`event`/`season_done`) |
| `src/historical_dataset/report.py` | **novo módulo** — `build_dataset_report()` / `write_dataset_report()` |
| `src/historical_dataset/cli.py` | **novo módulo** — `build_parser()`, `ProgressLogger`, `main(argv)` (implementação testável do CLI) |
| `build_historical_dataset.py` | passa a ser um wrapper fino sobre `src.historical_dataset.cli.main` |
| `.github/workflows/build_historical_dataset.yml` | **novo workflow**, `workflow_dispatch` apenas |

Nenhuma destas adições calcula odds, probabilidades, resultados ou
qualquer valor de modelo — são todas plumbing de execução/relatório sobre
dados já normalizados pelo builder existente.

---

## 2. Inputs do workflow (`workflow_dispatch`)

| Input | Tipo | Obrigatório | Omissão | Descrição |
|---|---|:---:|---|---|
| `competition_id` | string | não | (vazio = todas as ligas ativas) | ID da competição/liga na BSD API |
| `season_id` | string | não | (vazio = todas as épocas) | Restringe a uma única época |
| `output_format` | choice (`csv`, `sqlite`, `parquet`, `all`) | não | `all` | Formato(s) exportados |
| `page_size` | string | não | `100` | Tamanho de página nos pedidos paginados |
| `resume` | choice (`true`, `false`) | não | `false` | Ativa checkpoint/resume |

O workflow **só tem `workflow_dispatch`** — sem `schedule:` e sem trigger
em `push`/`pull_request`. Nenhuma execução automática é possível; tem de
ser disparada manualmente (UI do GitHub Actions, `workflow_dispatch` via
API, ou `gh workflow run`).

---

## 3. CLI (`build_historical_dataset.py` / `src.historical_dataset.cli`)

```bash
python build_historical_dataset.py \
  --competition-id 38 \
  --season-id 2025 \
  --output all
```

Flags relevantes para este workflow (ver `--help` para a lista completa,
incluindo as já existentes `--leagues`, `--country`, `--no-odds`,
`--no-stats`, `--odds-comparison`, `--rate-limit`, `--max-events`):

| Flag | Descrição |
|---|---|
| `--competition-id <id>` | Atalho para uma única competição (equivalente a `--leagues <id>`; não combinar com `--leagues`) |
| `--season-id <id>` | Restringe a construção a uma única época |
| `--output {csv,sqlite,parquet,all}` | Formato(s) a exportar |
| `--resume {true,false}` | Ativa checkpoint/resume (se `--checkpoint-dir` omitido, usa `<output-dir>/.checkpoint`) |
| `--page-size <n>` | Tamanho de página nos pedidos paginados |
| `--output-dir <dir>` | Diretório de saída (por omissão `data/historical`) |

### Exemplo de execução manual completa

```bash
export BSD_API_KEY=...   # ou BZZ_API_KEY / BZZOIRO_API_KEY / API_KEY, ou .env

python build_historical_dataset.py \
  --competition-id 38 \
  --season-id 2025 \
  --output all \
  --page-size 100 \
  --resume true \
  --output-dir data/historical
```

Saída esperada (exemplo ilustrativo — nenhuma execução real foi feita
nesta tarefa):

```
A construir dataset histórico a partir da BSD API...
[competição] Primeira Liga (id=38)
  [época] 2024/2025 (id=2025)
    [página 1] 100 jogos obtidos
    jogos processados: 25 | odds processadas: 24 | 3.10 jogos/s | ETA: 24s
    ...
  [época concluída] id=2025
Total de jogos processados: 306
Ficheiros exportados:
  csv: data/historical/historical_dataset.csv
  sqlite: data/historical/historical_dataset.sqlite
  parquet: data/historical/historical_dataset.parquet
Relatório de qualidade exportado para: data/historical/dataset_report.json
```

---

## 4. `dataset_report.json`

Gerado sempre (mesmo com 0 jogos), por `src.historical_dataset.report.build_dataset_report`
+ `write_dataset_report`, em `<output-dir>/dataset_report.json`:

```json
{
  "competition": 38,
  "season": 2025,
  "total_games": 306,
  "total_odds": 298,
  "duplicate_games": 0,
  "duplicate_odds": 0,
  "missing_values": {
    "overall_pct": 4.12,
    "by_column": { "odds_home": 2.61, "corners_home": 8.3, "...": "..." }
  },
  "execution_time_seconds": 187.4,
  "api_requests": 918,
  "output_files": {
    "csv": "data/historical/historical.csv",
    "sqlite": "data/historical/historical.sqlite",
    "parquet": "data/historical/historical.parquet"
  }
}
```

Definições exatas (ver docstrings em `src/historical_dataset/report.py`):

- `total_games` — nº de linhas do dataset exportado (um jogo por linha).
- `total_odds` — nº de jogos com pelo menos uma odd (1X2/Over-Under/BTTS) publicada.
- `duplicate_games` — nº de linhas com `event_id` repetido (deveria ser sempre 0 — o builder já deduplica; um valor > 0 indica dataset montado fora deste pipeline, ex. concatenação manual de exports).
- `duplicate_odds` — nº de linhas (entre as que têm odds) cujas equipas/data/odds coincidem exatamente com outra linha, sob `event_id`s diferentes — sinal de o mesmo jogo processado sob IDs diferentes.
- `missing_values` — `overall_pct` (% de células vazias em todo o dataset) e `by_column` (% de valores em falta por coluna).
- `execution_time_seconds` — tempo total da construção (medido pelo CLI, não pelo builder).
- `api_requests` — nº de pedidos HTTP feitos à BSD API nesta execução (`BSDHistoricalClient.request_count`).
- `output_files` — caminho de cada formato efetivamente escrito (só os pedidos por `--output`; `null` para Parquet se nenhum motor estiver instalado no ambiente).

---

## 5. Segurança — o que nunca é impresso

`ProgressLogger` (`src/historical_dataset/cli.py`) só recebe, do
`HistoricalDatasetBuilder`, IDs, nomes e contadores (`league_id`,
`league_name`, `season_id`, `season_name`, `page_number`, `items_count`,
`games_processed`, `odds_processed`) — nunca `client.api_key` nem os
headers HTTP enviados (`Authorization: Token ...`). O próprio
`dataset_report.json` também não contém nenhuma credencial.

No workflow, a chave é passada apenas via `env:` (nunca interpolada
diretamente numa string de `run:`) e o GitHub mascara automaticamente
qualquer valor de secret que apareça nos logs. Os inputs do
`workflow_dispatch` (`competition_id`, `season_id`, etc.) também são
passados via `env:` em vez de interpolados diretamente no script `bash`,
para evitar injeção de shell a partir de um input do workflow.

---

## 6. Artefactos

O último passo do workflow faz upload — **nunca commit** — dos ficheiros
que existirem em `data/historical/`:

```
historical.csv
historical.sqlite
historical.parquet
dataset_report.json
```

`if-no-files-found: ignore`: se `--output csv` foi usado, por exemplo, só
`historical.csv` e `dataset_report.json` existem — o upload não falha por
`historical.sqlite`/`historical.parquet` estarem ausentes. Os artefactos
ficam disponíveis na página da execução do workflow (aba *Actions* → a
execução → secção *Artifacts*, nome `historical-dataset-<run_id>`,
retenção de 30 dias) — não em nenhum branch/commit do repositório.

---

## 7. Como retomar (`--resume` / checkpoint)

Localmente, `--resume true` (com `--checkpoint-dir` explícito, ou o
omissão `<output-dir>/.checkpoint`) persiste em disco:

- `completed_seasons.json` — épocas totalmente concluídas.
- `processed_events.log` — jogos já processados (append-only).

Reexecutar o mesmo comando aponta para o mesmo diretório e salta
automaticamente tudo o que já foi concluído (ver
`docs/07_historical_dataset_builder.md`, secção "Checkpoint/resume").

**Limitação conhecida em GitHub Actions:** cada execução do
`workflow_dispatch` corre num runner efémero — o `data/historical/.checkpoint`
de uma execução não é preservado automaticamente para a execução seguinte
(o workflow não faz commit nem cache deste diretório, por desenho, para
não gravar dados no repositório). Nesta versão, `--resume true` no
workflow só é útil dentro da mesma execução (ex. um passo de shell que
corre o builder mais que uma vez) — para retomar entre execuções
separadas do workflow, correr localmente (onde o checkpoint persiste em
disco entre chamadas), ou adaptar o workflow para restaurar/gravar
`data/historical/.checkpoint` via `actions/cache` (fora do âmbito desta
tarefa — ver `docs/09_...` num trabalho futuro, se necessário).

---

## 8. Testes

`tests/historical_dataset/test_cli.py`, `test_report.py`,
`test_workflow_yaml.py`, mais extensões a `test_builder.py`,
`test_client.py`, `test_dedup.py` — todos com dados/clientes falsos, **sem
nenhuma chamada real à BSD API**. Cobrem: parsing de argumentos do CLI
(incluindo `--competition-id`/`--season-id`/`--output`/`--resume` e a
validação de conflito `--competition-id`+`--leagues`), geração de
`dataset_report.json`, estrutura do workflow (`workflow_dispatch` único,
sem `schedule`/`push`, inputs esperados, passo de upload com
`if-no-files-found: ignore`), e que nada imprime a chave de API.

```bash
python -m unittest discover -s tests/historical_dataset -v
python -m pytest tests/ -q   # suite completa do repositório
```

---

## 9. O que este trabalho explicitamente não fez

- Não chamou a BSD API (sem chave configurada nem rede permitida neste
  ambiente de desenvolvimento — ver `docs/BASELINE_RESULTS.md`).
- Não gerou nenhum dataset (`data/historical/*.csv/.sqlite/.parquet` não
  foram produzidos por este trabalho).
- Não usou dados sintéticos como substituto.
- Não alterou Dixon-Coles, Monte Carlo, λ estimator, Kelly, Edge, EV, Goal
  Engine, Decision Engine, Machine Learning, Backtesting Engine nem o
  Evaluation Framework.
