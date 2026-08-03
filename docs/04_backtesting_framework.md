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
    staking.py       # FlatStake / KellyStake (gestão de banca opcional)
    evaluator.py      # cálculo por aposta (edge/ev/kelly/stake/profit)
    metrics.py        # ROI, Yield, Drawdown, Profit Factor, etc.
    statistics.py      # Brier Score, Log Loss, ECE, calibração, distribuições
    segments.py        # análises por competição/mercado/odd/edge/ev/favorito/casa-fora
    thresholds.py       # threshold analysis (Edge/EV >= X%)
    report.py           # BacktestReport: tabela resumo, CSV, Excel, gráficos
    engine.py            # BacktestEngine — orquestrador / ponto de entrada único
    sample_data.py         # gerador de dataset sintético para demonstração/testes

tests/backtest/
    test_metrics.py, test_statistics.py, test_evaluator.py,
    test_thresholds_and_segments.py, test_integration.py
```

Coexiste com o `src/backtest/` já existente (`backtester.py`, `logger.py`,
`labeler.py`, `market.py`), que é específico da estratégia de pressão ao
vivo — este framework é genérico, para qualquer mercado/decisão histórica.

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
`venue`/`home_or_away`, `favorito`/`is_favorite`.

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

# dados: lista de dicts (aceita chaves PT/EN) ou pandas.DataFrame
engine = BacktestEngine(staking=FlatStake(unit=1.0))
report = engine.run(dados_historicos)

report.print_summary()               # tabela resumo no terminal
report.to_csv("output/backtest")     # bets.csv, summary.csv, segment_*.csv, ...
report.to_excel("output/report.xlsx")  # opcional, requer openpyxl
report.generate_all_plots("output/plots")  # equity curve, distribuições, calibração
```

Script de demonstração (dados sintéticos, ver secção seguinte):

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
    plots/
        equity_curve.png
        probability_distribution.png
        edge_distribution.png
        ev_distribution.png
        calibration_curve.png
```

---

## Testes

- **Unitários** (`tests/backtest/test_metrics.py`,
  `test_statistics.py`, `test_evaluator.py`,
  `test_thresholds_and_segments.py`): ROI, Yield, lucro líquido,
  drawdown, Brier Score, Log Loss, ECE, edge médio, EV médio, threshold
  analysis, segmentação — cada um com valores de referência calculados à
  mão.
- **Integração** (`tests/backtest/test_integration.py`): executa
  `BacktestEngine` de ponta a ponta sobre um dataset sintético de 60-150
  jogos (`sample_data.generate_sample_dataset`), incluindo exportação
  para CSV, Excel e gráficos.

Correr apenas os testes deste módulo:

```
python -m unittest discover -s tests/backtest -v
```

Nenhum teste pré-existente foi alterado; a suite completa
(`python -m unittest discover -s tests -v`) continua a passar na
íntegra após esta adição.
