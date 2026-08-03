# Auditoria Técnica — Erro 401 Unauthorized na Integração BSD Sports API

**Data:** 2026-08-03
**Âmbito:** Auditoria exaustiva da integração BSD Sports API e diagnóstico do erro 401 em `daily_engine_analysis.yml`.
**Regra seguida:** Nenhum algoritmo, modelo estatístico ou lógica de negócio foi alterado. Esta auditoria é apenas diagnóstica.

> **Estado: RESOLVIDO.** `BzzoiroClient` (`src/api/client.py`) foi corrigido para enviar
> `Authorization: Token <API_KEY>` em vez de `X-API-Key`, alinhando com o esquema `tokenAuth`
> da especificação (`schema.yaml`). Ver PR de correção subsequente a esta auditoria.
> Os achados abaixo mantêm-se como registo histórico do diagnóstico.

---

## 1. Mapa completo da integração BSD

| Ficheiro | Papel | Cliente HTTP usado |
|---|---|---|
| `src/config/settings.py` | Configuração central: `API_KEY`, `BASE_URL`, `BSD_ROOT_URL`, `require_api_key()` | — |
| `src/api/http_retry.py` | Helper genérico de retry/backoff (`get_with_retry`) usado por **ambas** as famílias de clientes | — |
| `src/api/client.py` | **`BzzoiroClient`** — cliente genérico de leitura (`GET {BASE_URL}/{endpoint}`) | Header `X-API-Key` |
| `src/api/live_fetcher.py` | `BSDLiveFetcher` — wrapper de `BzzoiroClient` para eventos ao vivo | via `BzzoiroClient` |
| `src/collector/client.py` | `EventCollector` — usado por `main.py predict` | via `BzzoiroClient` |
| `src/collector/odds.py` | `OddsCollector` — odds 1x2, usado por `EventCollector` | via `BzzoiroClient` |
| `src/collector/live_fetcher.py` | `LiveDataCollector` — parser de payloads estilo API-Football (não faz pedidos HTTP; não tem cliente próprio) | n/a |
| `src/live/providers/api_match_provider.py` | `APIMatchProvider` — jogos ao vivo (`main.py live`) | Header `Authorization: Token` |
| `src/live/providers/api_odds_provider.py` | `APIOddsProvider` — odds ao vivo (`main.py live`) | Header `Authorization: Token` |
| `src/live/providers/stats_provider.py` | `StatsProvider` — estatísticas de evento | Header `Authorization: Token` |
| `src/live/providers/incidents_provider.py` | `IncidentsProvider` — incidentes (golos, cartões) | Header `Authorization: Token` |
| `src/live/providers/bsd_feature_adapter.py` | `BSDFeatureAdapter` — transforma incidentes em features (sem HTTP) | n/a |
| `src/engine/live_pipeline.py` | `LivePipeline` — agrega `APIMatchProvider` + `APIOddsProvider` | via providers acima |
| `src/engine/predict_today.py` | **Código morto** — script standalone com a sua própria `fetch_enriched_data_from_bsd()` | via `BzzoiroClient` (mas não é chamado por nada em produção) |
| `src/engine/live_monitor.py` | Entry point do `live_logger.yml` | via `BzzoiroClient` (`BSDLiveFetcher`) |
| `scripts/app.py` | Dashboard Streamlit (`main.py dashboard`) | via `BzzoiroClient` (`BSDLiveFetcher`) |
| `scripts/live_scanner.py` | Script standalone de scan ao vivo | via `APIMatchProvider` (Token) |
| `research/pressure_shots/api.py` | Cliente de investigação isolado, com cache em disco (explicitamente marcado "não é código de produção") | Header `Authorization: Token` |
| `.github/workflows/daily_engine_analysis.yml` | Corre `python main.py predict` diariamente | → `BzzoiroClient` |
| `.github/workflows/live_logger.yml` | Corre `python src/engine/live_monitor.py` | → `BzzoiroClient` |
| `schema.yaml` | **Especificação OpenAPI 3.0.3 oficial da API** ("Bzzoiro Sports Data API"), incluída no repositório | Fonte de verdade sobre autenticação |
| `README.md` | Documentação do projeto | — |

---

## 2. Clientes duplicados

Existem **duas famílias de cliente HTTP independentes e incompatíveis** para a mesma API:

### Família A — `BzzoiroClient` (`src/api/client.py`)
- Único ponto de construção do pedido; todos os outros módulos desta família (`EventCollector`, `OddsCollector`, `BSDLiveFetcher`) delegam nele.
- Usa `requests` via `get_with_retry` (sessão *stateless*, sem `requests.Session`).
- **Header:** `X-API-Key: <key>`
- **Em produção, ativamente usada por:**
  - `python main.py predict` (via `EventCollector`) → workflow `daily_engine_analysis.yml`
  - `src/engine/live_monitor.py` (via `BSDLiveFetcher`) → workflow `live_logger.yml`
  - `python main.py dashboard` (via `BSDLiveFetcher`, `scripts/app.py`)

### Família B — `APIMatchProvider` / `APIOddsProvider` / `StatsProvider` / `IncidentsProvider` (`src/live/providers/`)
- Cada provider constrói o seu próprio dicionário de headers de forma independente e repetida (4 implementações quase idênticas).
- Usa `requests` via `get_with_retry` também (mesmo helper de retry).
- **Header:** `Authorization: Token <key>`
- **Em produção, ativamente usada por:**
  - `python main.py live` (via `LivePipeline`)
  - `scripts/live_scanner.py` (script standalone)
- **Não é usada por nenhum dos dois workflows agendados** (`daily_engine_analysis.yml`, `live_logger.yml`).

### Clientes mortos / isolados
- `src/engine/predict_today.py` — módulo completo, com a sua própria função `fetch_enriched_data_from_bsd()`, **não é importado por `main.py` nem por qualquer workflow**. Foi substituído por `src/cli/predict.py` + `EventCollector` no refactor do entrypoint único. Confirmado por `grep` — o único consumidor é o próprio módulo (`if __name__ == "__main__"`).
- `research/pressure_shots/api.py` — cliente de investigação isolado, com cache em disco, explicitamente documentado no próprio ficheiro como não sendo código de produção.
- Ficheiros `.bak` / `.bak2` (não executáveis, extensão inválida para import): `src/engine/live_pipeline.py.bak`, `src/report/dashboard.py.bak`, `src/report/dashboard.py.bak2`, `src/live/providers/api_match_provider.py.bak2`, `src/live/features/pressure.py.bak`, `src/live/engine.py.bak`, `.github/workflows/live_logger.yml.bak`. Não afetam a execução, mas são resíduos que nenhum ficheiro referencia.

**Nenhuma das duas famílias usa `requests.Session()`** — ambas fazem pedidos "soltos" via `requests.request(...)` dentro de `get_with_retry`/`post_with_retry`.

---

## 3. Auditoria de autenticação por cliente

| Cliente | Base URL | Endpoint exemplo | Método | Header enviado | Timeout | Retries | Tratamento de erro |
|---|---|---|---|---|---|---|---|
| `BzzoiroClient` | `https://sports.bzzoiro.com/api/v2` | `events/?limit=10` | GET | `X-API-Key: <key>`, `Accept: application/json` | 30s | 5 tentativas, backoff exponencial (`http_retry.py`) só para 429/5xx/timeout/connection error | `response.raise_for_status()` → propaga `requests.HTTPError` |
| `APIMatchProvider` | `https://sports.bzzoiro.com` (+ `/api/v2/...` manual no path) | `/api/v2/events/live/` | GET | `Authorization: Token <key>` | 10s | idem (via `get_with_retry`) | `r.raise_for_status()` |
| `APIOddsProvider` | idem | `/api/v2/events/{id}/odds/` | GET | `Authorization: Token <key>` | 10s | idem | `raise_for_status()` |
| `StatsProvider` | idem | `/api/v2/events/{id}/stats/` | GET | `Authorization: Token <key>` | 10s | idem | `raise_for_status()` |
| `IncidentsProvider` | idem | `/api/v2/events/{id}/incidents/` | GET | `Authorization: Token <key>` | 10s | idem | `raise_for_status()`, mas sem `try/except` no `.get("incidents", [])` |
| `research/pressure_shots/api.py` (isolado) | `https://sports.bzzoiro.com/api/v2` | `events/`, `events/{id}/stats/`, etc. | GET | `Authorization: Token <key>` | 30s | 3 tentativas manuais + cache em disco | `raise BzzoiroError` em 4xx não-429 |

**Fonte da chave em todos os casos:** `src/config/settings.py` — `BSD_API_KEY` → `BZZ_API_KEY` → `BZZOIRO_API_KEY` → `API_KEY` (primeira definida vence). Idêntico para as duas famílias.

---

## 4. `python main.py predict` vs `live_logger.yml` — usam o mesmo cliente?

**Sim, exatamente o mesmo cliente e a mesma configuração** (Família A, `X-API-Key`):

- `main.py predict` → `src/cli/predict.py::run_predict()` → `EventCollector()` → `BzzoiroClient()`
- `live_logger.yml` → `python src/engine/live_monitor.py` → `BSDLiveFetcher()` → `BzzoiroClient()`

Ambos usam a mesma `BASE_URL`, o mesmo header (`X-API-Key`), o mesmo `require_api_key()` e o mesmo helper de retry. **Não há divergência entre estes dois workflows** — a divergência real está entre esta Família A (usada pelos dois workflows agendados) e a Família B (`Authorization: Token`, usada apenas por `main.py live`, que não corre em nenhum workflow agendado).

---

## 5. Origem do erro 401 — evidências

### Evidência direta do log do CI (run `30809091479`, 2026-08-03T11:20:33Z)

```
Erro ao executar 'predict': 401 Client Error: Unauthorized for url: https://sports.bzzoiro.com/api/v2/events/?limit=10
```

Isto confirma: (a) o pedido chegou a ser feito (não é erro de rede/timeout), (b) os 4 env vars (`BSD_API_KEY`, `BZZ_API_KEY`, `BZZOIRO_API_KEY`, `API_KEY`) estavam presentes e mascarados no log como `***` (GitHub só mascara valores não-vazios), confirmando que o secret **está a ser carregado** — descarta a hipótese B.

### Evidência decisiva: especificação oficial da API (`schema.yaml`, incluído no repo)

```yaml
securitySchemes:
  tokenAuth:
    type: apiKey
    in: header
    name: Authorization
    description: 'Token-based auth. Format: "Token YOUR_API_KEY". Register free at /register/'
security:
- tokenAuth: []
```

Este esquema (`tokenAuth`) é o **único** mecanismo de autenticação definido em todo o `schema.yaml` (aplicado globalmente e repetido explicitamente em cada endpoint, incluindo `/api/v2/events/`). **Não existe nenhum `X-API-Key` security scheme na especificação.** Este padrão corresponde ao `TokenAuthentication` padrão do Django REST Framework.

`BzzoiroClient` (Família A) envia:
```python
headers={"X-API-Key": self.api_key, "Accept": "application/json"}
```

— um header que a API, segundo a sua própria especificação, **não reconhece como credencial**. Um pedido sem um cabeçalho `Authorization` válido é tratado como não-autenticado pelo DRF, resultando em `401 Unauthorized` — exatamente o comportamento observado.

Em contraste, os providers da Família B (`APIMatchProvider`, etc.) enviam:
```python
headers={"Authorization": f"Token {self.api_key}"}
```
— que corresponde **exatamente** ao formato exigido pela especificação.

### Avaliação das hipóteses

| Hipótese | Veredito | Evidência |
|---|---|---|
| A) Secret inválida/expirada | **Não é possível excluir isoladamente, mas não é a causa primária** | Não há nenhum pedido histórico bem-sucedido com este key+header para comparar (ver secção 5.1); mas o header errado já seria suficiente para gerar 401 mesmo com key válida |
| B) Secret não carregada | **Descartada** | Log do CI mostra os 4 aliases mascarados (`***`), i.e., não-vazios; `require_api_key()` não levantou `MissingAPIKeyError` (esse erro tem mensagem diferente e não gera pedido HTTP) |
| **C) Header incorreto** | **CONFIRMADA — causa primária** | `schema.yaml` da própria API define `Authorization: Token <key>` como único esquema válido; `BzzoiroClient` envia `X-API-Key` |
| D) API mudou o método de autenticação | **Sem evidência** | `schema.yaml` está datado de 2026-07-29 (poucos dias antes do incidente) e é consistente com o formato `Authorization: Token` usado desde sempre pela Família B |
| E) Endpoint mudou | **Descartada** | `/api/v2/events/` existe tal e qual na especificação, com os parâmetros usados (`limit`, `offset`, etc.) |
| **F) Clientes diferentes a comportar-se de forma diferente** | **CONFIRMADA — causa estrutural** | Duas famílias de cliente coexistem no mesmo repositório com headers de autenticação incompatíveis; a automação de produção (`predict`, `live_logger`) usa a família errada (`X-API-Key`) enquanto `main.py live` usa a família correta (`Authorization: Token`) |
| G) Outro motivo | Não aplicável | — |

### 5.1 Nota sobre o histórico do CI

Não existe, no histórico de execuções do workflow `daily_engine_analysis.yml`, **nenhuma execução anterior bem-sucedida** com o `main.py` atual (pós-refactor argparse, commits `7182582`/`d435e64`/`806e8f7`, todos de 2026-08-03). As execuções anteriores falharam por razões completamente diferentes e anteriores neste mesmo dia (`ImportError: cannot import name 'evaluate_decision'` em 2026-07-30; `RuntimeError: BSD_API_KEY missing` em 2026-08-01, causado pelo bug do `main.py` pré-refactor que executava sempre `LivePipeline()` independentemente do subcomando pedido — o mesmo bug já documentado no README). **Esta é, portanto, a primeira vez que o código chega efetivamente a fazer o pedido HTTP à API** com o `main.py predict` correto. Por isso não há، neste repositório, prova histórica de que a chave atual alguma vez tenha sido aceite pela API com o header `X-API-Key` nem com o header `Authorization: Token`.

**Nível de confiança da conclusão: Alto.**
A causa imediata e mecanicamente suficiente do 401 é o header de autenticação incorreto (`X-API-Key` em vez de `Authorization: Token`), confirmado pela especificação OpenAPI oficial da própria API, incluída no repositório. Não é possível excluir a 100% que a chave em si também esteja inválida (não há forma de testar isso sem fazer uma chamada real com o header correto), mas o header errado já é, por si só, causa suficiente e comprovada do erro observado.

---

## 6. Coerência entre README, `settings.py` e workflows

| Item | README | `settings.py` | Workflows | Coerente? |
|---|---|---|---|---|
| Nomes das env vars da API key | `BSD_API_KEY`, `BZZ_API_KEY`, `BZZOIRO_API_KEY`, `API_KEY` (nesta ordem) | `SUPPORTED_API_KEY_ENV_VARS` idêntico, mesma ordem | Os 4 são definidos com o mesmo valor (`secrets.BZZOIRO_API_KEY`) nos dois workflows | ✅ Sim |
| Endpoints documentados | Lista apenas os endpoints da Família B (`/events/live/`, `/events/{id}/`, `/odds/`, `/stats/`, `/incidents/`) | — | `BzzoiroClient`/`EventCollector` usam `events/?limit=`, `odds/?event_id=` — **não documentados no README** | ❌ Não — README documenta só metade da integração real |
| "Live Providers" (secção do README) | Lista `APIMatchProvider`, `APIOddsProvider`, `StatsProvider`, `IncidentsProvider`, `BSDFeatureAdapter` | — | Omite completamente `BzzoiroClient`, `EventCollector`, `OddsCollector`, `BSDLiveFetcher` — que são os clientes realmente usados pela automação agendada | ❌ Não |
| Descrição de `live_logger.yml` | "`.github/workflows/live_logger.yml` follows the same convention for its own subcommand" (implicando `python main.py <subcomando>`) | — | O workflow real corre `python src/engine/live_monitor.py` diretamente, sem passar por `main.py` | ❌ Não — descrição desatualizada/incorreta |
| Esquema de autenticação | Não menciona explicitamente qual header é usado | Não define header (fica em cada cliente) | Dois headers diferentes em uso | ❌ Não documentado em lado nenhum |

**Conclusão:** README, `settings.py` e workflows estão coerentes quanto às variáveis de ambiente, mas o README **não reflete a arquitetura real de produção** — documenta a família de clientes que não é usada pela automação agendada, e omite a que efetivamente falha.

---

## 7. Código legado identificado (apenas documentado, nada removido)

- **`src/engine/predict_today.py`** — módulo morto completo (função `fetch_enriched_data_from_bsd`, `main()`), substituído pelo par `src/cli/predict.py` + `src/collector/client.py`. Ainda referenciado por `src/tools/test_full_engine.py` (teste) e `research/backtest_engine.py`, `research/pressure_shots/predict_engine_bridge.py` (investigação), mas não por nenhum entry point de produção.
- **Ficheiros `.bak`/`.bak2`** (7 no total, listados na secção 2) — resíduos de refactors anteriores, sem extensão `.py` válida para import, inertes.
- **`research/pressure_shots/api.py`** — cliente de investigação isolado (auto-documentado como não sendo de produção), usa corretamente `Authorization: Token`, mas nunca partilhado com o código de produção — é uma terceira implementação paralela do mesmo conceito.
- **Header antigo em uso ativo (não apenas legado):** `X-API-Key` em `BzzoiroClient` — não é um "resíduo morto", é o header ativamente enviado pelos dois workflows de produção, e é o que está incorreto face à especificação atual da API.

---

## 8. Conclusões e recomendação técnica

### Arquitetura atual da integração BSD

```
                         ┌─────────────────────────────┐
                         │   src/config/settings.py     │
                         │ API_KEY (4 aliases de env)    │
                         │ BASE_URL / BSD_ROOT_URL        │
                         └───────────┬──────────────────┘
                                     │
              ┌──────────────────────┼───────────────────────┐
              │                                              │
   FAMÍLIA A — BzzoiroClient                    FAMÍLIA B — providers/*
   Header: X-API-Key  (❌ incorreto)             Header: Authorization: Token (✅ correto)
              │                                              │
  ┌───────────┼────────────┐                     ┌───────────┼───────────┐
  │           │            │                     │           │           │
EventCollector│      BSDLiveFetcher      APIMatchProvider APIOddsProvider ...
  │      OddsCollector      │                     │
  │           │             │                     │
main.py predict      src/engine/live_monitor.py   main.py live
  │           │             │                     │
daily_engine_        live_logger.yml         (não corre em
analysis.yml         (workflow)               nenhum workflow
(workflow)                                     agendado)
```

### Clientes ativos
- `BzzoiroClient` (`X-API-Key`) — usado por `main.py predict`, `live_monitor.py`/`live_logger.yml`, `main.py dashboard`.
- `APIMatchProvider`/`APIOddsProvider`/`StatsProvider`/`IncidentsProvider` (`Authorization: Token`) — usado por `main.py live`, `scripts/live_scanner.py`.

### Clientes obsoletos/mortos
- `src/engine/predict_today.py` (módulo inteiro, incluindo o seu próprio cliente inline).
- 7 ficheiros `.bak`/`.bak2`.
- `research/pressure_shots/api.py` é isolado mas não morto (usado por scripts de investigação).

### Causa provável do erro 401
`BzzoiroClient` envia o header `X-API-Key`, que **não corresponde** ao esquema de autenticação `tokenAuth` (`Authorization: Token <key>`) definido na especificação OpenAPI oficial da API (`schema.yaml`, incluída no repositório). É este o cliente usado por ambos os workflows de produção agendados.

### Nível de confiança
**Alto.** A evidência vem da própria especificação da API incluída no repositório, cruzada com o log de erro real do CI e com a segunda família de clientes que já usa o formato correto.

### Recomendação técnica (não implementada nesta fase)
Alinhar o header de `BzzoiroClient` com o esquema `tokenAuth` da especificação: `Authorization: Token <key>` em vez de `X-API-Key: <key>`. Isto **não é uma alteração de algoritmo, modelo ou lógica de negócio** — é uma correção de um cabeçalho HTTP de autenticação para o valor definido pela própria API. Adicionalmente, considerar (em PR separado, fora do âmbito desta auditoria):
- Unificar as duas famílias de cliente num único cliente HTTP para evitar recorrência deste tipo de divergência.
- Atualizar o README para refletir os clientes realmente usados pela automação (`BzzoiroClient`/`EventCollector`/`BSDLiveFetcher`) e corrigir a descrição de `live_logger.yml`.
- Remover ou arquivar `src/engine/predict_today.py` e os ficheiros `.bak`/`.bak2` depois de confirmação do utilizador.

### Ficheiros que terão de ser alterados no próximo PR (correção do 401)
- `src/api/client.py` — header `X-API-Key` → `Authorization: Token`.
- (Opcional, fora do âmbito imediato do 401) `README.md` — secção "Live Providers" e descrição de `live_logger.yml`.
