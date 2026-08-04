# Framework de Avaliação Quantitativa (`src/evaluation/`)

**Âmbito:** módulo puramente de MEDIÇÃO. Não altera, recalcula nem
substitui nenhum modelo matemático de previsão do Football Edge Engine —
Poisson, Dixon-Coles, Monte Carlo, estimativa de λ, Kelly, Edge/EV, Goal
Engine, Machine Learning e o Decision Engine permanecem exatamente como
estavam antes deste módulo. O objetivo é responder objetivamente à
pergunta "o sistema está a melhorar?", não prever resultados.

---

## 1. Auditoria — o que já existia vs. o que faltava

Antes de criar este módulo, o repositório já continha um Backtesting
Framework maduro em `src/backtest/historical/` (ver
`docs/04_backtesting_framework.md`), com testes próprios e integração com
`run_backtest.py`. A auditoria confirmou:

### Já implementado (reutilizado, não duplicado)

| Métrica / capacidade | Localização original |
|---|---|
| ROI, Yield, Profit, Hit Rate, Odd média, EV médio, Edge médio, Kelly médio, Stake total, Nº de apostas, Drawdown máximo, Profit Factor, Expectativa por aposta | `src/backtest/historical/metrics.py` |
| Brier Score, Log Loss, Calibration Error (ECE), curva de calibração, distribuições de probabilidade/edge/ev | `src/backtest/historical/statistics.py` |
| Segmentação por competição, mercado, faixa de odds, faixa de Edge, faixa de EV, favorito/underdog, casa/fora | `src/backtest/historical/segments.py` |
| Threshold analysis (Edge/EV ≥ X%) | `src/backtest/historical/thresholds.py` |
| Exportação CSV, Excel, HTML e gráficos (banca, distribuições, curva de calibração) | `src/backtest/historical/report.py` |
| CLI `run_backtest.py --demo` / `--input` | `run_backtest.py` |

### Em falta (implementado por este módulo)

| Requisito | Onde foi adicionado |
|---|---|
| Segmentação por **mês** | `src/evaluation/segments.py::segment_by_month` |
| Segmentação por **faixa de confiança** | `src/evaluation/segments.py::segment_by_confidence_range` |
| **Comparação objetiva entre duas execuções** de backtest (quem lucrou mais, maior ROI, menor Brier, melhor calibração) | `src/evaluation/compare.py` |
| **Relatório em Markdown** | `src/evaluation/report.py::EvaluationReport.to_markdown` |
| Gráfico de **ROI acumulado (%)** | `src/evaluation/plots.py::plot_cumulative_roi` |
| Gráfico de **Banca** (a partir de uma banca inicial configurável) | `src/evaluation/plots.py::plot_bankroll` |
| Gráfico de **distribuição de odds** | `src/evaluation/plots.py::plot_odds_distribution` |
| Gráfico de **lucro por competição** | `src/evaluation/plots.py::plot_profit_by_competition` |
| Gráfico de **yield por mercado** | `src/evaluation/plots.py::plot_yield_by_market` |
| **Reliability diagram** como export separado (nomenclatura pedida explicitamente) | `src/evaluation/plots.py::plot_reliability_diagram` |
| Módulo único, centralizado, que expõe tudo o que precede | `src/evaluation/__init__.py` |

Todas as capacidades "já implementadas" são reutilizadas por importação
direta (`from src.backtest.historical.metrics import ...`) — nunca
reimplementadas — para garantir que os números produzidos por
`src/evaluation/` são idênticos aos já validados pelos testes existentes
em `tests/backtest/`.

---

## 2. Localização

```
src/evaluation/
    __init__.py       # API pública: evaluate(), EvaluationReport, compare_backtests(), full_summary(), all_segments()
    metrics.py          # ROI, Yield, Profit, Hit Rate, Brier, LogLoss, ECE, avg odd/EV/Edge, stake total, n_bets
    segments.py           # todos os segmentos + by_month e by_confidence_range (novos)
    compare.py              # comparação objetiva entre dois ficheiros/execuções de backtest
    plots.py                  # ROI acumulado, banca, odds, lucro/competição, yield/mercado, reliability diagram
    report.py                   # EvaluationReport (CSV/Excel/HTML/Markdown) + evaluate()
    formatting.py                  # utilitário de tabela Markdown (sem depender do pacote 'tabulate')

tests/evaluation/
    test_metrics.py, test_segments.py, test_compare.py, test_report.py, test_plots.py
```

Coexiste com `src/backtest/historical/`, que continua a ser o motor de
cálculo oficial (Backtesting Framework). `src/evaluation/` é a camada de
avaliação/relatório que o consome — não o substitui.

---

## 3. Métricas implementadas

### 3.1 Financeiras (calculadas sobre as apostas **colocadas**, `placed_bets`)

| Métrica | Fórmula | Função |
|---|---|---|
| **ROI (%)** | lucro líquido total ÷ total apostado × 100 (ponderado pelo stake) | `evaluation.metrics.roi` |
| **Yield (%)** | média das rentabilidades individuais (`profit_i / stake_i`) × 100 (não ponderada pelo stake) | `evaluation.metrics.yield_pct` |
| **Profit** | soma de `profit` por aposta | `evaluation.metrics.net_profit` |
| **Hit Rate (%)** | apostas ganhas ÷ total de apostas × 100 | `evaluation.metrics.hit_rate` |
| **Expected Value médio (%)** | média de `ev` (fração) × 100 | `evaluation.metrics.avg_ev_pct` |
| **Odd média** | média de `odd` | `evaluation.metrics.avg_odd` |
| **Stake total** | soma de `stake` | `evaluation.metrics.total_staked` |
| **Número de apostas** | contagem de linhas | `evaluation.metrics.n_bets` |

Ver também `docs/04_backtesting_framework.md` para a distinção completa
entre ROI (ponderado pelo stake) e Yield (não ponderado) — relevante
sempre que o staking não é flat (ex. Kelly).

### 3.2 Estatísticas / calibração (calculadas sobre **todas** as apostas avaliadas, `all_bets`)

Medem a qualidade do modelo (o quão bem calibradas estão as
probabilidades previstas), independentemente de a aposta ter sido
colocada ou não.

| Métrica | Fórmula | Função |
|---|---|---|
| **Brier Score** | média((probabilidade prevista − resultado real)²). 0 = perfeito, 1 = pior possível | `evaluation.metrics.brier_score` |
| **Log Loss** | entropia cruzada binária, −média(y·log(p) + (1−y)·log(1−p)) | `evaluation.metrics.log_loss` |
| **Calibration Error (ECE)** | média ponderada (por contentor) da diferença absoluta entre probabilidade prevista e frequência real de acerto | `evaluation.metrics.calibration_error` |

### 3.3 Ponto de entrada único

```python
from src.evaluation.metrics import full_summary

metrics = full_summary(placed_df, all_df=all_bets_df)
# {..., "roi_pct": ..., "yield_pct": ..., "net_profit": ..., "hit_rate_pct": ...,
#  "avg_odd": ..., "avg_ev_pct": ..., "total_staked": ..., "n_bets": ...,
#  "brier_score": ..., "log_loss": ..., "calibration_error": ..., ...}
```

`full_summary` é apenas a união de `summary_metrics` (financeiras) e
`statistical_summary` (estatísticas) — nenhum valor é recalculado.

---

## 4. Análise segmentada

`src.evaluation.segments.all_segments(placed_bets)` devolve um dicionário
`{nome: DataFrame de métricas}` com uma linha por valor/faixa, cobrindo
todas as dimensões pedidas:

| Dimensão | Chave | Faixas por omissão |
|---|---|---|
| Competição | `by_competition` | um grupo por competição |
| Mercado | `by_market` | um grupo por mercado |
| Faixa de odds | `by_odd_range` | `1.00-1.50`, `1.50-2.00`, `2.00-3.00`, `3.00-5.00`, `5.00+` |
| Faixa de Edge | `by_edge_range` | `<0%`, `0-3%`, `3-5%`, `5-7%`, `7-10%`, `10-15%`, `15%+` |
| Faixa de confiança | `by_confidence_range` | `<50%`, `50-60%`, `60-70%`, `70-80%`, `80-90%`, `90-100%` |
| Mês | `by_month` | um grupo por "AAAA-MM", ordenado cronologicamente |

mais os segmentos adicionais herdados do Backtesting Framework:
`by_ev_range`, `by_favorite_vs_underdog`, `by_home_away`.

**Confiança REAL do modelo (Melhoria #8, `docs/AUDIT_MATEMATICA.md` §20):**
quando o dataset traz os metadados opcionais `lambda_tier` /
`effective_sample_size` / `model_confidence` (propagados de
`LambdaEstimate`, via `src.historical_dataset.backtest_bridge.lambda_confidence_from_dixon_coles`
+ `to_backtest_frame`), aparecem mais três segmentos —
`by_lambda_tier`, `by_effective_sample_size_range`,
`by_model_confidence` — cada um combinando ROI/Yield/Nº de apostas
(sobre as apostas colocadas do grupo) com Brier Score/Log Loss (sobre
todas as apostas avaliadas do grupo). Distinto de `by_confidence_range`
acima: aquele mede quão provável o modelo achou a SELEÇÃO apostada;
estes medem quanta INFORMAÇÃO sustentava a estimativa de λ que produziu
essa probabilidade. Omitidos automaticamente (sem erro) quando o dataset
não traz o metadado — ex. `examples/backtest/sample_real_games.csv`.

**Nota sobre "confiança":** o dataset de backtest não tem um campo de
confiança separado da probabilidade — `probability` (a probabilidade que
o modelo atribuiu à seleção apostada, já produzida pelo motor) é o único
valor de confiança por aposta disponível, e é o mesmo valor usado na
curva de calibração. Por isso "faixa de confiança" é implementada sobre
`probability`; nenhum conceito novo é introduzido no motor.

Cada linha de cada segmento é uma chamada normal a `full_summary`/
`summary_metrics` sobre o subconjunto de apostas desse grupo — mesma
fórmula, sem exceções por segmento.

---

## 5. Comparação entre versões

`src.evaluation.compare.compare_backtests` compara dois ficheiros
`bets.csv` (o export produzido por `to_csv(...)`, com uma linha por
aposta avaliada) — ou dois DataFrames já carregados — e responde às
quatro perguntas pedidas:

```python
from src.evaluation import compare_backtests

resultado = compare_backtests(
    "output/v1/bets.csv", "output/v2/bets.csv",
    label_a="v1", label_b="v2",
)

print(resultado.summary())
# {
#   "qual_modelo_ganhou_mais": "v2",
#   "qual_teve_maior_roi": "v2",
#   "qual_teve_menor_brier": "v1",
#   "qual_teve_melhor_calibracao": "v1",
# }

resultado.comparison_table()   # DataFrame lado-a-lado com todas as métricas de full_summary
resultado.to_markdown()        # relatório de comparação pronto a gravar em .md
```

Critérios: maior `net_profit` (lucro), maior `roi_pct`, menor
`brier_score`, menor `calibration_error` (ECE). Em caso de empate exato,
o critério devolve `"EMPATE"`. Métricas financeiras usam apenas as
apostas colocadas (`placed`); métricas estatísticas usam sempre o
ficheiro completo.

---

## 6. Relatórios gerados automaticamente

`EvaluationReport` (devolvido por `src.evaluation.evaluate(...)`) gera os
quatro formatos pedidos, todos a partir dos mesmos dados — nenhum formato
recalcula nada, apenas formata o que já foi calculado:

```python
from src.evaluation import evaluate
from src.backtest.historical.dataset import load_historical_dataset

dados = load_historical_dataset("examples/backtest/sample_real_games.csv")
report = evaluate(dados)

report.print_summary()
report.to_csv("output/evaluation")                  # bets.csv, summary.csv, segment_*.csv (inclui by_month, by_confidence_range), ...
report.to_excel("output/evaluation/report.xlsx")     # um único workbook, uma folha por secção
report.generate_all_plots("output/evaluation/plots") # 11 PNG (5 do Backtesting Framework + 6 novos)
report.to_html("output/evaluation/report.html", plots_dir="output/evaluation/plots")
report.to_markdown("output/evaluation/report.md")    # NOVO formato
```

---

## 7. Gráficos

| Gráfico | Ficheiro | Fonte |
|---|---|---|
| Evolução da banca (lucro acumulado absoluto) | `equity_curve.png` | já existia (`BacktestReport`) |
| Distribuição de probabilidades | `probability_distribution.png` | já existia |
| Distribuição de Edge | `edge_distribution.png` | já existia |
| Distribuição de EV | `ev_distribution.png` | já existia |
| Curva de Calibração | `calibration_curve.png` | já existia |
| **ROI acumulado (%)** | `cumulative_roi.png` | novo |
| **Banca** (a partir de banca inicial configurável) | `bankroll.png` | novo |
| **Distribuição de odds** | `odds_distribution.png` | novo |
| **Reliability Diagram** | `reliability_diagram.png` | novo (matematicamente idêntico à Curva de Calibração — exportado com este nome porque ambos os termos são pedidos explicitamente) |
| **Lucro por competição** | `profit_by_competition.png` | novo |
| **Yield por mercado** | `yield_by_market.png` | novo |

---

## 8. Uso

### 8.1 API Python

```python
from src.evaluation import evaluate
from src.backtest.historical import FlatStake, KellyStake
from src.backtest.historical.dataset import load_historical_dataset

dados = load_historical_dataset("meus_jogos.csv")
report = evaluate(dados, staking=FlatStake(unit=1.0), starting_bankroll=1000.0)

report.global_metrics          # ROI, Yield, Profit, Hit Rate, Odd média, Stake total, nº de apostas, ...
report.statistical_metrics     # Brier Score, Log Loss, ECE
report.all_segment_tables()    # {"by_competition": df, ..., "by_month": df, "by_confidence_range": df}
report.to_csv("output/eval")
report.to_excel("output/eval/report.xlsx")
report.to_html("output/eval/report.html", plots_dir="output/eval/plots")
report.to_markdown("output/eval/report.md")
report.generate_all_plots("output/eval/plots")
```

### 8.2 Comparar duas execuções

```python
from src.evaluation import compare_backtests

v1 = evaluate(dados_periodo_1)
v2 = evaluate(dados_periodo_2)
v1.to_csv("output/v1")
v2.to_csv("output/v2")

resultado = compare_backtests("output/v1/bets.csv", "output/v2/bets.csv", label_a="v1", label_b="v2")
print(resultado.summary())
```

### 8.3 Testes

```bash
python -m pytest tests/evaluation/ -v      # apenas este módulo
python -m pytest tests/ -q                 # suite completa do repositório
python run_backtest.py --demo              # confirma que o CLI e o Backtesting Framework originais continuam intactos
```

---

## 9. Testes

`tests/evaluation/` cobre:

- **`test_metrics.py`** — os wrappers de conveniência (`avg_odd`,
  `avg_ev_pct`, `avg_edge_pct`, `n_bets`) e `full_summary` produzem
  exatamente os mesmos valores que `summary_metrics`/`statistical_summary`
  originais.
- **`test_segments.py`** — `segment_by_month` e
  `segment_by_confidence_range` com valores calculados à mão; `all_segments`
  inclui todas as dimensões pedidas.
- **`test_compare.py`** — `compare_backtests` identifica corretamente o
  "vencedor" em cada um dos quatro critérios, incluindo o caso de empate.
- **`test_report.py`** — `evaluate(...)` de ponta a ponta sobre um dataset
  sintético (150 jogos), incluindo as quatro exportações e os segmentos
  extra.
- **`test_plots.py`** — séries auxiliares (`cumulative_roi_series`,
  `bankroll_series`) com valores calculados à mão, e todos os geradores de
  gráficos novos.

Nenhum teste pré-existente foi alterado; `python -m pytest tests/ -q`
continua a passar na íntegra (258 testes) após esta adição, e
`python run_backtest.py --demo` produz exatamente o mesmo resultado que
antes deste módulo (nenhuma fórmula do Backtesting Framework nem do motor
de previsão foi tocada).

---

## 10. O que este módulo não faz

- Não altera Poisson, Dixon-Coles, Monte Carlo, λ, Kelly, Edge, EV, Goal
  Engine, Machine Learning ou o Decision Engine.
- Não decide quando apostar — apenas mede o que já aconteceu.
- Não substitui `src/backtest/historical/` — depende dele para todas as
  fórmulas financeiras/estatísticas já validadas.
