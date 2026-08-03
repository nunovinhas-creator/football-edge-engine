# Backtesting Framework

**Âmbito:** módulo novo, adicionado após a correção da definição oficial
de Edge (`src/engine/edge.py`). Não altera nenhuma fórmula matemática do
motor — Poisson, Dixon-Coles, Monte Carlo, Goal Engine, Kelly, Edge, EV e
Machine Learning permanecem exatamente como estavam. O objetivo é medir,
não prever.

---

## Localização

```
src/backtest/historical/
    models.py       # HistoricalBet / EvaluatedBet (contrato de dados)
    dataset.py         # carregamento/normalização de jogos históricos (CSV) para o BacktestEngine
    staking.py       # FlatStake / KellyStake (gestão de banca opcional)
    evaluator.py      # cálculo por aposta (edge/ev/kelly/stake/profit)
    metrics.py        # ROI, Yield, Drawdown, Profit Factor, etc.
    statistics.py      # Brier Score, Log Loss, ECE, calibração, distribuições
    segments.py        # análises por competição/mercado/odd/edge/ev/favorito/casa-fora
    thresholds.py       # threshold analysis (Edge/EV >= X%)
    report.py           # BacktestReport: tabela resumo, CSV, Excel, HTML, gráficos
    engine.py            # BacktestEngine — orquestrador / ponto de entrada único
    sample_data.py         # gerador de dataset sintético para demonstração/testes

run_backtest.py              # CLI: backtest completo / por data / competição / mercado
examples/backtest/
    sample_real_games.csv      # pequeno conjunto de jogos históricos REAIS (ver secção "Dataset")

tests/backtest/
    test_metrics.py, test_statistics.py, test_evaluator.py,
    test_thresholds_and_segments.py, test_integration.py,
    test_dataset.py, test_dataset_e2e.py
```

Coexiste com o `src/backtest/` já existente (`backtester.py`, `logger.py`,
`labeler.py`, `market.py`), que é específico da estratégia de pressão ao
vivo — este framework é genérico, para qualquer mercado/decisão histórica.

---

## Dataset de jogos históricos (`dataset.py`)

`src/backtest/historical/dataset.py` carrega jogos históricos e prepara-os
para o `BacktestEngine` **sem alterar nenhum algoritmo de previsão** —
consome apenas o que já foi produzido pelo motor (probabilidade, odd,
decisão), ou dados que decorrem diretamente do resultado real do jogo:

- `load_historical_dataset(source, ...)` — ponto de entrada único. Aceita
  um caminho para CSV, um `DataFrame` ou uma lista de dicts, e devolve um
  `DataFrame` normalizado pronto para `BacktestEngine.run(...)`. Aceita
  aliases em português ou inglês para todas as colunas (tal como
  `HistoricalBet.from_dict`).
- `filter_dataset(df, start_date=, end_date=, competition=, market=)` —
  filtra o dataset já normalizado; usado pelo CLI `run_backtest.py`.
- `infer_market_result(market, home_goals, away_goals)` — deriva
  "WIN"/"LOSS" para HOME/DRAW/AWAY, OVER_X.X/UNDER_X.X e BTTS a partir do
  resultado final do jogo (golos casa/fora); é apenas uma tabela de
  mapeamento sobre um resultado já ocorrido, não uma previsão.

O projeto ainda não tem uma integração própria com uma fonte de jogos
históricos com odds e resultados (as fontes existentes — `src.api`,
`src.collector` — servem eventos futuros/ao vivo, não um arquivo
histórico); por isso a via suportada é **CSV**, com o seguinte esquema
mínimo:

| Coluna (EN)    | Alias (PT)                        | Obrigatória | Descrição                                             |
|-----------------|-------------------------------------|:-----------:|---------------------------------------------------------|
| `date`          | `data`                              | sim          | data do jogo                                              |
| `competition`   | `competicao`, `liga`, `league`        | sim          | competição                                                  |
| `home_team`     | `equipa_casa`, `casa`                  | sim          | equipa da casa                                               |
| `away_team`     | `equipa_visitante`, `visitante`, `fora`  | sim          | equipa visitante                                              |
| `market`        | `mercado`                              | sim          | mercado recomendado (ex. `HOME`, `OVER_2.5`)                    |
| `odd`           | `odd_disponivel`, `bookie_odd`           | sim          | odd disponível no momento da previsão                             |
| `model_prob`    | `probabilidade`, `prob_model`             | sim          | probabilidade prevista pelo motor (fração 0.0-1.0)                  |
| `home_goals`    | `golos_casa`                                | não\*        | golos da equipa da casa no final do jogo                             |
| `away_goals`    | `golos_fora`                                | não\*        | golos da equipa visitante no final do jogo                            |
| `result`        | `resultado`                                  | não\*        | resultado final do mercado ("WIN"/"LOSS"), se já conhecido               |
| `engine_decision` | `decisao`                                  | não          | decisão histórica do motor ("BET"/"PASS"/"WAIT"), se já conhecida           |

\* É necessário fornecer `result`/`resultado` OU `home_goals`+`away_goals`
(o resultado do mercado é derivado automaticamente do resultado final via
`infer_market_result` quando `result` está ausente).

Quando `engine_decision` está ausente, é preenchida chamando o
`DecisionEngine` real (`src.engine.decision`) sobre `model_prob` e `odd` —
a mesma decisão que o motor de previsão já produziria para esses valores.
Edge, EV e Kelly **nunca** são calculados em `dataset.py`: continuam a ser
calculados exclusivamente por `evaluator.py`, a partir de
`src.engine.edge` / `src.engine.kelly`, tal como no resto do framework.

---

## Contrato de dados de entrada

Cada linha histórica (dict, DataFrame ou `HistoricalBet`) deve conter, no
mínimo (aceita chaves em português ou inglês):

| Campo (PT)              | Campo (EN)        | Descrição                                    |
|--------------------------|--------------------|-----------------------------------------------|
| `jogo`                   | `match`            | identificador do jogo                          |
| `data`                   | `date`             | data do jogo                                    |
| `mercado`                | `market`           | mercado da aposta                                |
| `odd`                    | `odd`              | odd decimal disponível (> 1.0)                    |
| `probabilidade`          | `model_prob`       | probabilidade do modelo, fração (0.0-1.0]           |
| `decisao`                | `engine_decision`  | decisão histórica do motor ("BET"/"PASS"/"WAIT")     |
| `resultado`               | `result`           | "WIN"/"LOSS" (ou bool/1-0)                            |

Campos opcionais para segmentação: `competicao`/`competition`,
`equipa_casa`/`home_team`, `equipa_visitante`/`away_team`,
`venue`/`home_or_away`, `favorito`/`is_favorite`. Quando `home_team` e
`away_team` estão presentes (é o caso quando os dados vêm de
`dataset.load_historical_dataset`), `match`/`jogo` é derivado
automaticamente como `"<home_team> vs <away_team>"` se não for fornecido
explicitamente, e ambos os campos são preservados em `EvaluatedBet` (logo,
também nos CSV/Excel/HTML exportados por `BacktestReport`).

## Contrato de saída

`BacktestEngine.run(dados)` devolve um `BacktestReport` com:

- `all_bets` — todas as apostas avaliadas (probabilidade, probabilidade de
  mercado, edge, ev, kelly, stake hipotético, resultado, lucro líquido).
- `placed_bets` — subconjunto de `all_bets` onde `engine_decision`
  continha "BET" (é sobre este subconjunto que o ROI/Yield são medidos).
- `global_metrics` — ROI, Yield, Profit, nº de apostas, hit rate, odd
  média, edge médio, EV médio, Kelly médio, drawdown máximo, profit
  factor, expectativa por aposta.
- `statistical_metrics` — Brier Score, Log Loss, ECE (calculados sobre
  **todas** as apostas avaliadas, não só as colocadas — medem a
  calibração do modelo, não a rentabilidade da estratégia).
- `calibration_curve`, `segments` (por competição/mercado/odd/edge/ev/
  favorito-underdog/casa-fora) e `edge_thresholds` / `ev_thresholds`
  (threshold analysis).

### ROI vs. Yield

- **ROI** = lucro líquido total ÷ total apostado × 100 (ponderado pelo
  stake — reflete o retorno sobre o capital investido).
- **Yield** = média das rentabilidades individuais (`profit_i / stake_i`)
  × 100 (não ponderada pelo stake). Coincide com o ROI quando o stake é
  fixo (flat staking); diverge quando o stake varia (ex. Kelly), pois uma
  aposta grande e perdedora domina o ROI mas pesa o mesmo que qualquer
  outra no Yield.

---

## Uso

```python
from src.backtest.historical import BacktestEngine, FlatStake, KellyStake
from src.backtest.historical.dataset import load_historical_dataset, filter_dataset

# 1. Carregar jogos históricos (CSV, DataFrame ou lista de dicts — ver secção "Dataset")
dados = load_historical_dataset("examples/backtest/sample_real_games.csv")

# 2. (Opcional) Filtrar por data/competição/mercado
dados = filter_dataset(dados, start_date="2015-01-01", competition="Champions League")

# 3. Correr o BacktestEngine (sem tocar em nenhuma fórmula do motor)
engine = BacktestEngine(staking=FlatStake(unit=1.0))
report = engine.run(dados)

report.print_summary()                     # tabela resumo no terminal
report.to_csv("output/backtest")           # bets.csv, summary.csv, segment_*.csv, ...
report.to_excel("output/backtest/report.xlsx")  # opcional, requer openpyxl
report.generate_all_plots("output/backtest/plots")  # equity curve, distribuições, calibração
report.to_html("output/backtest/report.html", plots_dir="output/backtest/plots")  # relatório HTML autocontido
```

### CLI (`run_backtest.py`)

```bash
# Backtest completo sobre um CSV de jogos históricos
python run_backtest.py --input meus_jogos.csv

# Backtest por intervalo de datas
python run_backtest.py --input meus_jogos.csv --start-date 2015-01-01 --end-date 2020-12-31

# Backtest por competição
python run_backtest.py --input meus_jogos.csv --competition "Premier League"

# Backtest por mercado
python run_backtest.py --input meus_jogos.csv --market OVER_2.5

# Demo rápida com o pequeno conjunto de jogos históricos REAIS incluído no repositório
python run_backtest.py --demo
```

Cada execução produz, em `--output-dir` (por omissão `output/backtest`):
`bets.csv`, `summary.csv`, `segment_*.csv`, `edge_thresholds.csv`,
`ev_thresholds.csv`, `backtest_report.xlsx`, `backtest_report.html` e
`plots/*.png`. Ver `python run_backtest.py --help` para todas as opções
(staking flat/Kelly, banca, etc.).

Script de demonstração com dados totalmente sintéticos (ver secção seguinte):

```
python -m src.tools.run_backtest_example [output_dir] --n-games 500 --seed 42
```

---

## Exemplo de execução

Executado sobre **500 apostas históricas sintéticas** geradas por
`src.backtest.historical.sample_data.generate_sample_dataset` — o
repositório não contém um dataset histórico real, pelo que este exemplo
serve para demonstrar o funcionamento do framework, não para tirar
conclusões sobre o modelo real. Os dados sintéticos simulam
deliberadamente um modelo levemente sobreconfiante (probabilidade do
modelo = probabilidade real + ruído positivo), para que a Threshold
Analysis e a curva de calibração tenham algo de interessante para mostrar.

Das 500 apostas geradas, o `DecisionEngine` real (`src/engine/decision.py`)
sinalizou 148 como "BET".

### Resumo global (apostas colocadas)

| Métrica              | Valor      |
|-----------------------|------------|
| Nº de apostas          | 148        |
| Taxa de acerto         | 46.62%     |
| Total apostado         | 148.0      |
| Lucro líquido          | -10.69     |
| ROI                    | -7.22%     |
| Yield                  | -7.22%     |
| Odd média              | 2.24       |
| Edge médio             | 6.67%      |
| EV médio               | 15.23%     |
| Kelly médio            | 14.77%     |
| Drawdown máximo        | -29.36 (-340.6%) |
| Profit Factor          | 0.86       |
| Expectativa por aposta | -0.07      |

### Métricas estatísticas (todas as 500 apostas avaliadas)

| Métrica            | Valor   |
|----------------------|---------|
| Brier Score           | 0.2236  |
| Log Loss              | 0.6386  |
| Calibration Error (ECE) | 0.0567 |

### Threshold Analysis (Edge)

| Edge ≥ | Nº apostas | Hit Rate | ROI     | Yield   | Lucro  |
|--------|------------|----------|---------|---------|--------|
| 1%     | 226        | 46.02%   | -9.09%  | -9.09%  | -20.54 |
| 3%     | 148        | 46.62%   | -7.22%  | -7.22%  | -10.69 |
| 5%     | 93         | 50.54%   | 3.32%   | 3.32%   | 3.09   |
| 7%     | 52         | 53.85%   | **19.94%** | **19.94%** | 10.37 |
| 10%    | 20         | 45.00%   | 0.70%   | 0.70%   | 0.14   |
| 15%    | 3          | 33.33%   | 6.33%   | 6.33%   | 0.19   |

Neste exemplo sintético, o melhor threshold por ROI com amostra razoável
(≥ 20 apostas) é **Edge ≥ 7%**: menos apostas, mas taxa de acerto e ROI
claramente superiores ao conjunto completo — exatamente o tipo de
descoberta empírica que este módulo se destina a produzir. Com dados
reais, os números (e o threshold ótimo) serão diferentes; o processo é o
que importa.

### Segmentos (excerto)

Favorito vs. Underdog:

| Tipo      | Apostas | Hit Rate | ROI     |
|-----------|---------|----------|---------|
| FAVORITE  | 74      | 63.51%   | -0.69%  |
| UNDERDOG  | 74      | 29.73%   | -13.76% |

Casa vs. Fora:

| Venue | Apostas | Hit Rate | ROI     |
|-------|---------|----------|---------|
| HOME  | 86      | 48.84%   | -1.62%  |
| AWAY  | 62      | 43.55%   | -15.00% |

(Tabelas completas — por competição, mercado, intervalo de odds, edge e
EV — nos ficheiros `segment_*.csv` gerados por `report.to_csv(...)`.)

### Ficheiros gerados pelo exemplo

```
output_dir/
    bets.csv                  # uma linha por aposta avaliada
    summary.csv                # métricas globais + estatísticas
    calibration_curve.csv        # curva de calibração (bins)
    edge_thresholds.csv           # threshold analysis (Edge)
    ev_thresholds.csv              # threshold analysis (EV)
    segment_by_*.csv                # uma tabela por segmento
    backtest_report.xlsx              # tudo o que precede, num único Excel
    backtest_report.html               # o mesmo conteúdo, em HTML autocontido (gráficos embutidos)
    plots/
        equity_curve.png
        probability_distribution.png
        edge_distribution.png
        ev_distribution.png
        calibration_curve.png
```

---

## Exemplo com jogos históricos reais (`--demo`)

`examples/backtest/sample_real_games.csv` contém um pequeno conjunto (8 jogos)
de resultados históricos **reais e verificáveis publicamente** — datas,
equipas e resultados finais verídicos (ex. Man Utd 1-6 Man City em
23/10/2011, Barcelona 5-0 Real Madrid em 29/11/2010, a final da Champions
League de 2019). As odds e a probabilidade do modelo associadas a cada
jogo são ilustrativas — o repositório não contém um arquivo de odds
históricas nem as probabilidades reais que o motor teria produzido nesse
momento — servindo apenas para demonstrar e testar o pipeline completo de
ponta a ponta (`run_backtest.py --demo`, e
`tests/backtest/test_dataset_e2e.py`).

```bash
python run_backtest.py --demo
```

```
A carregar dataset histórico de: examples/backtest/sample_real_games.csv
Jogos carregados: 8
Jogos após filtragem: 8
========================================
BACKTEST — RESUMO DE DESEMPENHO
========================================
n_bets              : 3
wins                : 2
losses              : 1
hit_rate_pct        : 66.67
...
```

Este exemplo real também permite exercitar os três modos de filtragem do
CLI sobre o mesmo dataset:

```bash
python run_backtest.py --demo --start-date 2017-01-01 --end-date 2019-12-31  # 3 jogos
python run_backtest.py --demo --competition "Champions League"                # 4 jogos
python run_backtest.py --demo --market OVER_2.5                                # 2 jogos
```

---

## Testes

- **Unitários** (`tests/backtest/test_metrics.py`,
  `test_statistics.py`, `test_evaluator.py`,
  `test_thresholds_and_segments.py`, `test_dataset.py`): ROI, Yield, lucro
  líquido, drawdown, Brier Score, Log Loss, ECE, edge médio, EV médio,
  threshold analysis, segmentação, derivação do resultado do mercado e
  normalização/filtragem do dataset histórico — cada um com valores de
  referência calculados à mão.
- **Integração** (`tests/backtest/test_integration.py`): executa
  `BacktestEngine` de ponta a ponta sobre um dataset sintético de 60-150
  jogos (`sample_data.generate_sample_dataset`), incluindo exportação
  para CSV, Excel e gráficos.
- **End-to-end com jogos reais** (`tests/backtest/test_dataset_e2e.py`):
  carrega `examples/backtest/sample_real_games.csv` (8 jogos históricos reais),
  corre o pipeline completo — carregamento, filtragem por data/competição/
  mercado, `BacktestEngine.run` e exportação para CSV/Excel/HTML/gráficos —
  e valida os resultados de mercado esperados a partir dos resultados
  finais reais desses jogos.

Correr apenas os testes deste módulo:

```
python -m unittest discover -s tests/backtest -v
```

Nenhum teste pré-existente foi alterado; a suite completa
(`python -m unittest discover -s tests -v`) continua a passar na
íntegra após esta adição.
