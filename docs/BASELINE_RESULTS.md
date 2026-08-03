# Baseline Oficial do Football Edge Engine — Estado: BLOQUEADO

**Data da tentativa:** 2026-08-03
**Âmbito pedido:** avaliação quantitativa completa (Historical Dataset
Builder → Backtesting Framework → Evaluation Framework) sobre o maior
dataset histórico real possível suportado pela BSD API.
**Resultado desta execução:** a baseline **não foi produzida**. A
construção do dataset histórico real foi bloqueada antes do primeiro
pedido bem-sucedido à BSD API, por razões de acesso de ambiente/rede, não
de código. Por instrução explícita, **não foi gerada nenhuma baseline
sintética/demonstrativa como substituto** — este documento é apenas o
relatório técnico do bloqueio.

---

## 1. Bloqueio exato

Dois problemas independentes impedem qualquer pedido real à BSD API a
partir do ambiente de execução onde este trabalho foi tentado:

### 1.1 Sem chave de API disponível na sessão

`src/config/settings.py` procura a chave em, por esta ordem, `BSD_API_KEY`,
`BZZ_API_KEY`, `BZZOIRO_API_KEY`, `API_KEY` (variáveis de ambiente ou
`.env`). Nenhuma destas está definida no ambiente desta sessão — o único
sítio onde a chave real existe é o secret `BZZOIRO_API_KEY` do GitHub
Actions, injetado apenas dentro dos workflows `daily_engine_analysis.yml`
e `live_logger.yml` (ver `docs/AUDIT_BSD_401.md`), não disponível a uma
sessão de desenvolvimento interativa.

```
$ python3 -c "from src.config import settings; settings.require_api_key()"
MissingAPIKeyError: BSD API key not configured. ...
```

### 1.2 Rede bloqueada ao nível da política de saída do ambiente

Mesmo assumindo uma chave válida, o próprio acesso de rede a
`sports.bzzoiro.com` está a ser rejeitado antes de chegar à aplicação —
não é um 401/403 da API, é uma rejeição do proxy de saída do ambiente:

```
$ curl -o /dev/null -w "HTTP %{http_code}\n" https://sports.bzzoiro.com/api/v2/leagues/
HTTP 000   (curl exit code 56 — connection reset)

$ curl -sS "$HTTPS_PROXY/__agentproxy/status"
"recentRelayFailures": [
  {
    "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "sports.bzzoiro.com:443"
  }
]
```

`sports.bzzoiro.com` não consta da lista de destinos permitidos
(`noProxy`/allowlist) da política de rede deste ambiente. O gateway de
saída recusa o `CONNECT` TCP com 403 antes de qualquer handshake TLS ou
pedido HTTP — ou seja, isto acontece independentemente de existir ou não
uma chave de API válida.

### 1.3 Consequência

Como os dois bloqueios são independentes e ambos suficientes por si só,
**nenhum jogo, competição, época, odd ou estatística real foi obtido**
nesta tentativa. Não houve nenhum pedido HTTP bem-sucedido a
`sports.bzzoiro.com` durante este trabalho.

---

## 2. Confirmação — nenhum código/algoritmo foi alterado

Esta tentativa foi puramente de execução/diagnóstico. Verificação:

```
$ git status --porcelain
(vazio)

$ git diff --stat
(vazio)
```

A árvore de trabalho ficou inalterada durante toda a tentativa — nenhum
ficheiro em `src/engine/` (Dixon-Coles, Monte Carlo, λ estimator, Kelly,
Edge, EV, Goal Engine, Machine Learning, Decision Engine),
`src/historical_dataset/`, `src/backtest/` ou `src/evaluation/` foi
tocado. O único ficheiro novo é este próprio relatório
(`docs/BASELINE_RESULTS.md`).

Como confirmação adicional de que o repositório está saudável e nenhuma
fórmula foi tocada por este trabalho, a suite de testes completa
(pré-existente, sem alterações) foi executada — isto **não** requer
acesso à BSD API, porque todos os testes usam dados/clientes falsos:

```
$ python -m pytest tests/ -q
336 passed, 1 warning in 11.15s
```

Este resultado confirma o estado do repositório *antes* desta tentativa;
não constitui uma baseline nem substitui a avaliação quantitativa pedida.

---

## 3. O que falta para produzir a baseline real

1. Uma chave de API BSD válida, disponibilizada ao ambiente de execução
   (variável `BSD_API_KEY`, `BZZ_API_KEY`, `BZZOIRO_API_KEY` ou `API_KEY`,
   ou `.env` na raiz do repositório).
2. Acesso de rede de saída a `sports.bzzoiro.com:443` permitido pela
   política do ambiente (allowlist do proxy/gateway).

Sem (2), (1) sozinho não resolve o bloqueio — o pedido é rejeitado ao
nível do transporte, antes de qualquer autenticação.

---

## 4. Comando exato para retomar assim que o acesso estiver disponível

Depois de (1) e (2) resolvidos, a sequência completa pedida é a seguinte
— nenhum destes comandos requer alteração de código, todos já existem no
repositório tal como está:

```bash
# 0. Instalar dependências (se necessário)
pip install -r requirements.txt

# 1. Construir o maior dataset histórico possível suportado pela BSD API
#    (todas as ligas ativas, todas as épocas, com checkpoint/resume)
python build_historical_dataset.py \
  --output-dir data/historical \
  --checkpoint-dir data/historical/.checkpoint \
  --odds-comparison

# 2. Validar dimensão/qualidade do dataset (nº de jogos, competições, épocas,
#    mercados, cobertura de odds, % valores em falta, duplicados) — a partir
#    de data/historical/historical_dataset.csv (ou .sqlite/.parquet)

# 3. Converter para o formato do Backtesting Framework (mercado à escolha,
#    model_prob vindo do motor de previsão já existente, sem o alterar)
python -c "
from src.historical_dataset.backtest_bridge import to_backtest_frame
import pandas as pd
records = pd.read_csv('data/historical/historical_dataset.csv').to_dict('records')
# model_prob tem de vir de src.engine.* aplicado a estes jogos, não deste builder
bridge_df = to_backtest_frame(records, market='HOME', model_prob=minha_probabilidade_do_modelo)
bridge_df.to_csv('data/historical/backtest_input.csv', index=False)
"

# 4. Correr o Backtesting Framework sobre o dataset completo
python run_backtest.py --input data/historical/backtest_input.csv \
  --output-dir output/baseline/backtest

# 5. Correr o Framework de Avaliação Quantitativa (ROI, Yield, Brier, Log
#    Loss, ECE, drawdown, segmentação por competição/época/mercado/odds/
#    edge/confiança, CSV/Excel/HTML/Markdown, todos os gráficos)
python -c "
from src.evaluation import evaluate
from src.backtest.historical.dataset import load_historical_dataset
dados = load_historical_dataset('data/historical/backtest_input.csv')
report = evaluate(dados)
report.to_csv('output/baseline/evaluation')
report.to_excel('output/baseline/evaluation/report.xlsx')
report.generate_all_plots('output/baseline/evaluation/plots')
report.to_html('output/baseline/evaluation/report.html', plots_dir='output/baseline/evaluation/plots')
report.to_markdown('output/baseline/evaluation/report.md')
"

# 6. Repetir os passos 3-5 para os mercados 1X2 (HOME/DRAW/AWAY),
#    Over/Under e BTTS para a comparação entre mercados pedida, e
#    substituir este documento pelos resultados reais.
```

Ver `docs/07_historical_dataset_builder.md`, `docs/04_backtesting_framework.md`
e `docs/06_model_evaluation.md` para a documentação completa de cada
etapa (já existente, não alterada por este trabalho).

---

## 5. O que este documento explicitamente NÃO é

- **Não é uma baseline.** Nenhum número de ROI, Yield, Hit Rate, Brier
  Score, Log Loss, ECE, Drawdown ou lucro por segmento apresentado aqui —
  porque nenhum foi calculado sobre dados reais nem sintéticos.
- **Não usa dados sintéticos/demonstrativos como substituto**, por
  instrução explícita — `src/backtest/historical/sample_data.py` (gerador
  sintético) e `examples/backtest/sample_real_games.csv` (8 jogos de
  demonstração) não foram executados neste trabalho com essa finalidade.
- **Não altera** o Evaluation Framework, o Backtesting Framework, o
  Historical Dataset Builder, nem qualquer algoritmo do motor (Dixon-Coles,
  Monte Carlo, λ estimator, Kelly, Edge, EV, Goal Engine, Machine Learning,
  Decision Engine).

---

## 6. Recomendações

1. Confirmar com o proprietário do ambiente/infraestrutura se
   `sports.bzzoiro.com` pode ser adicionado à allowlist de saída de rede
   para sessões de desenvolvimento (ou executar este pipeline num ambiente
   com rede equivalente à dos GitHub Actions, onde os workflows já
   demonstram conseguir alcançar a API).
2. Disponibilizar a chave BSD (`BZZOIRO_API_KEY` ou equivalente) a esse
   mesmo ambiente, por `.env` ou variável de ambiente.
3. Depois de resolvidos (1) e (2), retomar exatamente pela secção 4 deste
   documento — nenhum código adicional precisa de ser escrito.
