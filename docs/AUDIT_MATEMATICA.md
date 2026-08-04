# Auditoria Matemática Completa — Football Edge Engine

**Data:** 2026-08-03
**Âmbito:** Auditoria estática, sem alterações de código. Objetivo: documentar exatamente como cada componente matemático do sistema funciona hoje, incluindo simplificações, inconsistências e bugs. Nenhuma otimização ou correção foi aplicada — apenas observação e registo.

**Convenção:** todas as referências usam o formato `caminho/ficheiro.py:linha`.

---

## Sumário Executivo

O Football Edge Engine **não é um único pipeline matemático coerente** — é uma coleção de **três a quatro pipelines paralelos e parcialmente sobrepostos**, escritos em momentos diferentes, que partilham nomes de conceitos (edge, EV, Kelly, confiança) mas têm implementações divergentes e, nalguns casos, incompatíveis entre si. Não existe um único caminho "BSD API → Feature Engineering → Probability Model → Poisson → Monte Carlo → Goal Engine → EV → Kelly → Decision Engine → Output" como está descrito no pedido de auditoria; existem **quatro entry points independentes** (`main.py live`, `main.py predict`, `main.py train`, `main.py dashboard`) que percorrem sub-conjuntos diferentes e incompletos dessa cadeia.

Achados mais importantes (detalhados nas secções seguintes):

1. **O modelo Dixon-Coles/Poisson (`src/engine/dixon_coles.py`) está matematicamente correto mas nunca é chamado por nenhum entry point de produção.** É código orfão — só é exercitado por testes.
2. **A simulação Monte Carlo em produção usa λ fixos "chumbados" (hard-coded), não os λ dinâmicos calculados pelo próprio sistema.** `src/report/dashboard.py:44-45` ignora completamente o `calculate_dynamic_lambda()` de `src/engine/live_pipeline.py:31-44` e o resultado dessa simulação "correta" (`pipeline_sim`) fica numa variável nunca usada (`dashboard.py:36`).
3. **O "Goal Engine" ao vivo (`src/live/engine.py`) não é um modelo de Poisson nem probabilístico — é um score heurístico aditivo, clampado a [0,100] e apresentado como percentagem de probabilidade.** Os pesos (0.60, 0.15, 0.25, 0.10, etc.) não têm justificação estatística documentada nem foram calibrados.
4. **O provider de dados ao vivo (`APIMatchProvider.get_live_match`) força a zero todas as métricas de pressão em tempo real** (`dangerous_attacks_10m=0`, `shots_on_target_10m=0`, `shots_10m=0`, `corners_10m=0`, `possession=50.0` — `src/live/providers/api_match_provider.py:128-137`), pelo que o Goal Engine, quando alimentado por este provider, funciona sempre com pressão = 0 nos componentes de volume de jogo.
5. **Existem duas fórmulas de EV inconsistentes ativas ao mesmo tempo**: `EV = p·odd − 1` (correto, maioria dos módulos) e um bug em `src/engine/market.py:23-26` que chama `calculate_edge(model_probability, market_probability)` — passando uma **probabilidade** (0–1) onde a função espera uma **odd decimal** (>1), o que produz valores de edge sistematicamente errados sempre que este módulo é usado.
6. **O modelo XGBoost "calibrado" que é efetivamente carregado no dashboard (`models_data/xgboost_live_v1.pkl`, via `src/model/ml_predictor.py`) foi treinado inteiramente com dados sintéticos gerados por `np.random` (`src/model/train.py:7-36`)** — nunca viu um jogo real. É rotulado na UI como "XGBoost Live v1.0 (Calibrated)", o que sugere fiabilidade que não existe.
7. Em paralelo, existe um **segundo pipeline de treino, muito mais rigoroso** (`src/training/train_model.py`) — com GroupKFold por `match_id`, deteção de leakage, calibração sigmoid e otimização de threshold por F1 — que treina sobre dados reais (`data/training_dataset.csv`, derivados de `data/live_history.db`). Mas o modelo que produz (`models/live_goal_model.pkl`) **não existe no repositório** e o predictor que o carregaria (`src/live/ml_predictor.py::MLGoalPredictor`) **não é importado por nenhum código de produção**. É trabalho de engenharia sólido que está desligado do sistema.
8. **A base de dados `live_history.db` tem três implementações diferentes e mutuamente inconsistentes da lógica de rotulagem `goal_in_next_15m`** (`src/backtest/logger.py::update_outcomes`, `src/training/create_labels.py`, `src/backtest/labeler.py`), com janelas temporais diferentes. Apenas duas correm em produção (via `live_logger.yml`), e a segunda sobrepõe-se à primeira em cada execução.
9. `src/engine/decision.py` define **a função `make_decision` duas vezes** (linhas 58 e 93) — em Python, a segunda definição sobrepõe-se silenciosamente à primeira, tornando a primeira implementação morta e enganadora para quem lê o ficheiro.
10. Não existe validação estatística formal (Brier score, log-loss, calibration plot) para nenhum modelo excepto o pipeline `train_model.py` — que, como referido, está desligado da produção.

Classificação de maturidade global: **protótipo de investigação com fragmentos de engenharia de produção séria, não um sistema quantitativo de produção coeso.**

---

## 1. Mapa da Pipeline Matemática (fluxo real vs. fluxo desenhado)

### 1.1 Fluxo pedido na auditoria (idealizado)

```
BSD API → Feature Engineering → Probability Model → Poisson → Monte Carlo →
Goal Engine → Expected Value → Kelly Criterion → Decision Engine → Output
```

### 1.2 Fluxo real encontrado no código

O sistema tem **4 entry points** em `main.py:10-15`, cada um com o seu próprio sub-grafo:

```
main.py
 ├── live      → src/cli/live.py::run_live()
 ├── predict   → src/cli/predict.py::run_predict()
 ├── train     → src/cli/train.py::run_train()
 └── dashboard → scripts/app.py (Streamlit, fora do âmbito desta leitura de código)
```

**Caminho `live` (o mais próximo do "Goal Engine" pedido):**

```
BSD API (events/{id}, stats, incidents)
   [src/live/providers/api_match_provider.py, stats_provider.py, incidents_provider.py]
        │  (dangerous_attacks_10m, shots_10m, shots_on_target_10m, corners_10m
        │   FORÇADOS a 0; possession forçado a 50.0 — ver §4.4)
        ▼
LiveMatchState (dataclass)  [src/models/live_state.py]
        │
        ├──► LiveGoalEngine.predict_next_goal_probability()  [src/live/engine.py:40]
        │       → pressure, dominance_index, estimated_xg_10m, next_goal_probability (heurística aditiva)
        │
        ├──► LivePipeline.calculate_dynamic_lambda()  [src/engine/live_pipeline.py:31]
        │       → λ_home dinâmico (baseado em live_xg e pressure)
        │       → λ_away = 0.80 HARD-CODED (live_pipeline.py:64)
        │
        ├──► MonteCarloSimulator.run_match_simulation()  [src/engine/simulation.py:17]
        │       → over_15, over_25, btts, xG esperado (resultado guardado em `analysis["simulation"]`
        │         mas descartado no dashboard — ver §1.3)
        │
        ▼
src/report/dashboard.py::render_live_dashboard()
        ├── volta a correr MonteCarloSimulator, agora com home_lambda=1.6 / away_lambda=1.1
        │   HARD-CODED (dashboard.py:44-45) — SUBSTITUI o resultado dinâmico anterior
        ├── DecisionEngine.evaluate_bet() [src/engine/decision.py:21] → Kelly + edge + ação
        ├── GoalWindowPredictor.predict_window() [src/live/features/goal_window.py:11]
        ├── LiveMLPredictor.predict() [src/model/ml_predictor.py:50] → modelo XGBoost sintético
        └── evaluate_live_market() [src/engine/live_decision.py:13] → edge/ação de mercado live
        ▼
Consola Rich (tabelas) — Output
```

O modelo **Dixon-Coles/Poisson bivariado** (`src/engine/dixon_coles.py`) e o `evaluate_match_value()` de `src/engine/value.py` **não aparecem em nenhum ponto deste grafo**. Apenas são chamados em testes (`src/tools/test_*.py`) e no ficheiro `.bak` `src/engine/live_pipeline.py.bak`, que não é executado.

**Caminho `predict` (pré-jogo, mercado 1X2):**

```
BSD API (events, odds)  [src/collector/client.py]
   ▼
predict_probability(event)  [src/model/predictor.py:1]
   → heurística baseada em H2H (não Poisson, não ML)
   ▼
implied_probability / calculate_edge / calculate_ev  [src/engine/edge.py]
   ▼
make_decision(edge, ev)  [src/engine/decision.py:93 — 2ª definição, ativa]
   ▼
calculate_stake(edge, confidence, h2h_matches)  [src/engine/stake.py:1]
   → NÃO é Kelly. É uma tabela heurística fixa (edge × 0.05 a 0.5)
   ▼
create_ranking() → print_report()  [src/engine/ranking.py, src/report/printer.py]
```

**Caminho `predict_today.py` (usado pelo workflow diário `daily_engine_analysis.yml`, mas não invocado por `main.py`):**

```
BSD API (events, odds, janela de 3 dias)  [src/engine/predict_today.py:15]
   ▼
RandomForestClassifier treinado INLINE, a cada execução, sobre
research/pressure_shots/features_v2.csv (dataset de REMATES, não de golos/1X2)
OU dados sintéticos [predict_today.py:108-126]
   ▼
model.predict_proba() → prob (probabilidade de "over 12.5 remates")
   ▼
run_pipeline(prob_model=prob, odd_house=odd_1X2_do_evento, ...)  [src/engine/full_engine.py:29]
   → edge/EV calculados comparando uma probabilidade de REMATES com uma odd de
     RESULTADO 1X2 do evento (ver §9.3 — mismatch de mercado)
   ▼
Telegram (boletim diário)
```

**Caminho `train`:**

```
main.py train → src/cli/train.py → src/model/train.py::train_and_save_model()
   → dataset 100% sintético (np.random, seed=42) → XGBoost + CalibratedClassifierCV
   → grava em models_data/xgboost_live_v1.pkl (o ficheiro efetivamente usado pelo dashboard)
```

Existe ainda um **quinto pipeline de treino**, mais robusto, que não tem entry point em `main.py` e corre apenas manualmente ou via GitHub Action (`live_logger.yml`, passo 4.5, que só recalcula labels — não treina):

```
data/live_history.db → src/training/build_dataset.py → data/training_dataset.csv
   → src/training/create_labels.py (rotula goal_in_next_15m)
   → src/training/train_model.py (GroupKFold, 6 modelos, calibração, threshold ótimo)
   → models/live_goal_model.pkl (NÃO existe no repositório — nunca foi corrido/commitado)
```

### 1.3 Conclusão da secção 1

O enunciado do pedido de auditoria descreve uma pipeline linear única. **Essa pipeline não existe como tal no código.** Existem sub-sistemas matemáticos genuínos (Dixon-Coles, Monte Carlo, Kelly, um pipeline de treino ML rigoroso) mas estão **desligados uns dos outros** — construídos, aparentemente, em iterações sucessivas do projeto sem que as anteriores tenham sido removidas ou integradas. Isto é a limitação estrutural mais relevante de toda a auditoria e afeta a interpretação de todas as secções seguintes: quando se pergunta "o sistema usa xG?" ou "o sistema usa Poisson?", a resposta correta é sempre **"sim, num módulo — mas não necessariamente no caminho que corre em produção."**

---

## 2. Probability Model

Não existe *um* "Probability Model". Existem quatro implementações de probabilidade base, nenhuma delas ligada a Poisson:

### 2.1 `src/model/predictor.py::predict_probability()` (usado por `main.py predict`)

Heurística pura baseada em confrontos diretos (H2H):

- Começa em `probability = 50` (linha 3).
- Soma **+3 fixo** por vantagem de jogar em casa (linha 36) — constante fixa, não calibrada, não depende da equipa nem da liga.
- Peso do histórico H2H por escalão de amostra: `total_matches ≥ 10 → peso 5`; `≥5 → 3`; `≥3 → 2`; `<3 → 1` (linhas 40-50).
- Se `home_win_rate > away_win_rate`: `probability += peso`; caso contrário `-= peso` (linhas 54-61).
- Ajuste por média de golos H2H: `+2` se `avg_goals ≥ 3`, `-2` se `avg_goals < 2` (linhas 66-73).
- Se `total_matches < 3`: regressão para a média, `probability = (probability + 50) / 2` (linhas 78-82).
- **Clamp final: `[35, 65]`** (linhas 88-93) — a probabilidade nunca pode sair deste intervalo, independentemente dos dados.

**Variáveis usadas:** apenas `head_to_head.total_matches`, `home_win_rate`, `away_win_rate`, `avg_total_goals`. Não usa xG, forma recente, lesões, odds, força ofensiva/defensiva, nem Poisson.

**Classificação:** heurística baseada em regras (rule-based), não estatística nem calibrada. O clamp `[35,65]` é uma proteção anti-extremos artificial, não uma calibração de probabilidade (Platt/Isotonic).

### 2.2 `src/model/train.py` + `src/model/ml_predictor.py::LiveMLPredictor` (usado por `main.py live`)

XGBoost + `CalibratedClassifierCV(method="sigmoid")` — **mas treinado exclusivamente em dados sintéticos** (`generate_synthetic_training_data`, `train.py:7-36`): minutos, ataques perigosos, remates, cantos, posse, pressão anterior — todos `np.random.randint`/`np.random.uniform` independentes entre si, e o alvo `y` gerado por uma função sigmoide artificial sobre um sinal sintético (`train.py:32-34`). Não há qualquer jogo real nos dados de treino.

Classificação formal: **calibrado** (tecnicamente — `CalibratedClassifierCV` foi aplicado), mas calibrado **em relação a uma distribuição sintética que não representa futebol real**. Em produção, o output é apresentado com `confidence_score=92.0` fixo (`model/ml_predictor.py:58`) sempre que o modelo carrega — um número de confiança que não deriva de nenhuma métrica, é uma constante.

Fallback (quando o `.pkl` não existe): heurística linear `(shots_on_target×0.25 + dangerous_attacks×0.05 + corners×0.08) × 10`, clampada a `[5,95]`, com `confidence_score=65.0` fixo (`ml_predictor.py:62-73`).

### 2.3 `src/live/engine.py::LiveGoalEngine.predict_next_goal_probability()` (o "Goal Engine")

Ver §5 — é um score heurístico aditivo, não uma probabilidade estatística. Auditado em detalhe na secção do Goal Engine.

### 2.4 `predict_today.py` / `predict_engine_bridge.py` — RandomForest sobre features de remates

`RandomForestClassifier(n_estimators=100, max_depth=5)` treinado ad-hoc, a cada execução, sobre `attack_avg_last5`, `dangerous_attack_avg_last5`, `ball_safe_avg_last5`, `total_shots_avg_last5`, `shots_on_target_avg_last5`, e as respetivas diferenças contra o adversário. Alvo: `total_shots > 12.5` (mercado de remates, não golos nem 1X2). **Estatístico e supervisionado** — é o único módulo desta secção que usa features reais (rolling averages de jogos anteriores) e treino/teste apropriados. Mas o output é depois usado como se fosse a probabilidade do mercado 1X2 sendo apostado (ver §9.3).

### 2.5 Resumo da classificação (Probability Model)

| Módulo | Calibrado | Heurístico | Estatístico | Híbrido | Usado em produção? |
|---|---|---|---|---|---|
| `model/predictor.py` (H2H) | Não | **Sim** | Não | — | Sim (`main.py predict`) |
| `model/ml_predictor.py` (XGBoost sintético) | Sim (formalmente), mas sobre dados falsos | — | Sim (formalmente) | — | Sim (`main.py live`) |
| `live/engine.py` (Goal Engine) | Não | **Sim** | Não | — | Sim (`main.py live`) |
| `predict_today.py` / `predict_engine_bridge.py` (RF remates) | Não | — | **Sim** | — | Sim (workflow diário), mas para mercado errado |
| `training/train_model.py` (pipeline rigoroso) | Sim | — | Sim | — | **Não** (artefacto não existe) |

---

## 3. Modelo de Poisson / Dixon-Coles

Ficheiro: `src/engine/dixon_coles.py`.

### 3.1 Fórmula

Distribuição de Poisson bivariada com correção de dependência de Dixon & Coles (1997) para resultados baixos:

```
P(X=x, Y=y) = Poisson(x; λ_home) · Poisson(y; μ_away) · τ(x,y)
```

com o fator de correção (`dixon_coles.py:4-17`):

```
τ(0,0) = 1 − λ_home·μ_away·ρ
τ(1,0) = 1 + μ_away·ρ
τ(0,1) = 1 + λ_home·ρ
τ(1,1) = 1 − ρ
τ(x,y) = 1   para todos os outros (x,y)
```

com `ρ = -0.05` por omissão (`dixon_coles_simulate_match(..., rho: float = -0.05, ...)`, linha 19). Este é exatamente o formato do paper original de Dixon-Coles. A matriz de resultados é normalizada no fim para somar 1 (linha 33), o que é matematicamente correto e necessário porque a correção τ não preserva a soma total de probabilidade exatamente.

Mercados 1X2 (`src/engine/value.py:8-14`):

```
P(home) = Σ (triângulo inferior da matriz, home_goals > away_goals)   [np.tril(matrix, -1)]
P(draw) = Σ (diagonal, home_goals == away_goals)                       [np.trace(matrix)]
P(away) = Σ (triângulo superior, home_goals < away_goals)              [np.triu(matrix, 1)]
```

Matematicamente correto e bem implementado.

### 3.2 Como é calculado λ?

**Este é o ponto crítico: `dixon_coles_simulate_match()` recebe `lambda_home` e `mu_away` como argumentos externos — o próprio ficheiro `dixon_coles.py` não calcula λ a partir de nada.** Não há, em nenhum ponto deste módulo, um cálculo de força ofensiva/defensiva por equipa, médias de liga, ou regressão de Poisson (o método clássico de Dixon-Coles estima λ_home = μ_liga × ataque_casa × defesa_fora × vantagem_casa, ajustado por MLE sobre dados históricos — nada disto está implementado).

Quem chama esta função em produção? **Ninguém.** `evaluate_match_value()` (`value.py`) e `dixon_coles_simulate_match()` só aparecem em:
- `src/tools/test_*.py` (testes unitários, com λ passados manualmente como floats de teste);
- `src/engine/live_pipeline.py.bak` (ficheiro `.bak`, não executado);
- `research/backtest_engine.py` (não chama Dixon-Coles diretamente — confundido por grep textual com "lambda" de Python).

O λ efetivamente usado em produção vem de dois sítios diferentes e desconectados:

1. **`LivePipeline.calculate_dynamic_lambda()`** (`src/engine/live_pipeline.py:31-44`):
   ```
   base_lambda = 1.20
   xg_factor = live_xg_10m × 0.30
   pressure_factor = (pressure / 100) × 0.60
   λ_home = min(base_lambda + xg_factor + pressure_factor, 4.0)
   ```
   com `λ_away = 0.80` fixo (linha 64). Este λ_home **é calculado mas nunca chega ao ecrã** — ver §1.2/§3.3.

2. **`src/report/dashboard.py:44-45`**: `home_lambda=1.6, away_lambda=1.1` — **constantes fixas**, independentes de qualquer input do jogo. É este par de valores que efetivamente alimenta o Monte Carlo mostrado ao utilizador.

### 3.3 Checklist de inputs pedida na auditoria

| Pergunta | Resposta |
|---|---|
| λ dinâmico? | Existe um cálculo dinâmico (`calculate_dynamic_lambda`), mas **não chega ao output final** (dashboard usa constantes 1.6/1.1). |
| Vantagem casa? | Não modelada explicitamente em Dixon-Coles (não há termo separado); em `model/predictor.py` existe um `+3` fixo aditivo (não multiplicativo, não específico por equipa/liga). |
| Usa xG? | `calculate_dynamic_lambda` usa `live_xg_10m` (estimado, não xG real de provider) — mas este λ é descartado antes do output (ver acima). O xG "histórico" (`home_xg_last5`, `away_conceded_xg_last5`) do `LiveMatchState` vem hard-coded em `api_match_provider.py` nalguns caminhos (1.65/1.30 em `live_fetcher.py:78-80`) e não reflete os últimos 5 jogos reais da equipa apesar do nome do campo. |
| Usa forma recente? | Não, no sentido de rolling stats de jogos passados dentro do módulo Dixon-Coles/Poisson. (O módulo de research de remates, `build_history_features.py`, sim, calcula rolling-5 — mas para outro mercado, ver §2.4.) |
| Usa lesões? | Não, em lado nenhum do sistema. |
| Usa odds? | Não no cálculo de λ. As odds só entram depois, no cálculo de EV/edge. |
| Usa força ofensiva/defensiva? | Não no sentido Dixon-Coles clássico (ataque_casa × defesa_fora). O único proxy é `xg_factor` e `pressure_factor` no λ dinâmico, que também não chega ao output. |

### 3.4 Validação matemática

- A fórmula de τ e a normalização estão **corretas** e correspondem à literatura (Dixon & Coles, 1997, *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*).
- `max_goals=8` (linha 19) é um truncamento razoável (P(gol>8) é despicienda para futebol).
- `ρ=-0.05` está dentro do intervalo tipicamente reportado na literatura para ligas europeias (normalmente entre -0.1 e 0), mas é uma constante fixa não estimada dos dados (não há MLE de ρ).
- **Simplificação estrutural mais importante:** o modelo não faz a segunda metade do método Dixon-Coles — a estimação de λ por equipa via regressão de Poisson com parâmetros de ataque/defesa e decaimento temporal (`ξ`, time-decay weighting). Só está implementada a "metade da frente" (a fórmula de likelihood/probabilidade dado λ), não a "metade de trás" (a calibração de λ a partir de dados históricos). Isto é consistente com o facto de o módulo nunca ser invocado em produção — não haveria de onde vir um λ calibrado.

---

## 4. Simulação Monte Carlo

Ficheiro: `src/engine/simulation.py`.

### 4.1 Mecânica

```python
remaining_ratio = max(0, (90 - current_minute) / 90)
rem_home_lambda = home_lambda × remaining_ratio
rem_away_lambda = away_lambda × remaining_ratio

simulated_home_goals = np.random.poisson(rem_home_lambda, n_simulations)
simulated_away_goals = np.random.poisson(rem_away_lambda, n_simulations)

final_home = current_home_score + simulated_home_goals
final_away = current_away_score + simulated_away_goals
```

- **Número de simulações:** `n_simulations=1000` por omissão (`simulation.py:14`), tanto em `LivePipeline` como em `render_live_dashboard` — nunca configurado a partir de contexto (ex.: não aumenta simulações quando a decisão é marginal).
- **Distribuição:** `numpy.random.poisson`, gerador Mersenne Twister do NumPy legacy (`np.random`, não o novo `Generator` API) — adequado para o propósito, sem controlo de seed (não reprodutível entre execuções).
- **Independência:** os golos de casa e fora são amostrados como **duas Poisson independentes** (não há correlação entre elas, ao contrário do Dixon-Coles do §3, que modela dependência para resultados baixos via τ). É uma escolha de simplificação consistente com "assumir independência condicional a λ", mas nota-se a inconsistência de haver, no mesmo repositório, um módulo (Dixon-Coles) que trata explicitamente esta dependência e outro (Monte Carlo) que a ignora — sem que nenhum dos dois se comunique com o outro de qualquer forma.
- **Convergência/estabilidade:** com N=1000 simulações, o erro-padrão de uma proporção estimada perto de p=0.5 é ≈ √(0.25/1000) ≈ 1.6 p.p.; para probabilidades mais extremas (p=0.1 ou p=0.9) o erro cai para ≈1.0 p.p. Isto é aceitável para decisões com thresholds de poucos pontos percentuais (o `DecisionEngine` usa `min_edge=5.0`, ver §8), mas introduz ruído de simulação da mesma ordem de grandeza que alguns dos limiares de decisão usados noutros módulos (ex.: `live_decision.py` usa limiares de edge de 3 e 10 pontos percentuais — o ruído de ±1-2 p.p. da simulação pode empurrar uma decisão de um lado do limiar para o outro em execuções diferentes do mesmo jogo).
- **Viés identificado:** ver §3.2/§4.2 — os λ usados na simulação mostrada ao utilizador são **constantes fixas** (1.6/1.1 no dashboard), não os λ ajustados ao tempo restante calculados corretamente pela fórmula `remaining_ratio`. O `remaining_ratio` em si está corretamente implementado (escala linear do tempo restante), mas ao ser aplicado sobre λ fixos, o resultado não reflete o jogo real que está a ser analisado — dois jogos completamente diferentes ao minuto 60 produziriam a mesma distribuição de golos restantes.

### 4.2 Duas simulações desconectadas no mesmo output

No `LivePipeline.evaluate()` (`live_pipeline.py:59-65`), a simulação corre com `home_lambda` dinâmico e `away_lambda=0.80` fixo, ignorando **completamente** o placar atual do jogo (`current_home_score=0, current_away_score=0` está hard-coded na chamada, linhas 61-62, independentemente do resultado real do `match_state`).

No `render_live_dashboard()` (`dashboard.py:40-46`), corre uma **segunda instância separada** de `MonteCarloSimulator`, desta vez com o placar real (`h_score, a_score` extraídos da string `score`) mas com λ completamente fixos (1.6/1.1).

Ou seja: **nenhuma das duas simulações usa simultaneamente (a) o placar real e (b) um λ que reflita o estado do jogo.** Uma tem o placar certo e λ errado; a outra tem o λ (nominalmente) dinâmico mas o placar errado (sempre 0-0).

### 4.3 Correção aplicada — Monte Carlo passa a consumir o λ dinâmico

**Antes:** `render_live_dashboard()` (`dashboard.py`) instanciava um segundo `MonteCarloSimulator` e chamava `run_match_simulation()` com `home_lambda=1.6` e `away_lambda=1.1` fixos, ignorando por completo o resultado já calculado por `LivePipeline.evaluate()` (que ficava na variável `pipeline_sim`, nunca usada — ver §4.2 e achado nº2 do sumário executivo).

**Agora:** existe uma única execução do Monte Carlo por avaliação, feita em `LivePipeline.evaluate()` (`src/engine/live_pipeline.py`). O dashboard deixou de instanciar `MonteCarloSimulator` e passou a consumir diretamente `analysis["simulation"]` — o resultado produzido com o λ_home dinâmico de `calculate_dynamic_lambda()`. Não foram alterados o algoritmo de simulação, o número de simulações, a distribuição usada (Poisson via `numpy.random.poisson`), o Decision Engine, o Kelly, o Expected Value, o Goal Engine ou qualquer componente de Machine Learning — apenas a origem dos valores de λ consumidos pela chamada já existente a `run_match_simulation()`.

- **λ_home:** dinâmico, único ponto de cálculo em `LivePipeline.calculate_dynamic_lambda()` (inputs: `estimated_xg_10m` e `pressure`, produzidos por `LiveGoalEngine.predict_next_goal_probability()`).
- **λ_away:** continua a não ter uma fonte dinâmica no sistema (não existe, atualmente, um sinal equivalente de pressão/xG específico da equipa visitante em `LiveMatchState`) — mantém-se o valor fixo que já era usado por `LivePipeline` antes desta alteração (`0.80`), agora nomeado como `FALLBACK_LAMBDA_AWAY` e documentado no código. Construir uma fonte dinâmica para λ_away ficaria fora do âmbito desta alteração (exigiria novos dados/lógica no Goal Engine).
- **Fallback do λ_home:** se `calculate_dynamic_lambda()` receber dados ausentes ou não numéricos (ex.: `live_result` incompleto), devolve `FALLBACK_LAMBDA_HOME = 1.6` — o mesmo valor fixo que o dashboard usava incondicionalmente antes desta correção — em vez de propagar uma excepção. A simulação nunca falha por falta de dados ao vivo.
- **Testes:** `tests/test_live_pipeline.py` (λ dinâmico reflete os inputs ao vivo; fallback ativado com dados ausentes/inválidos; `evaluate()` nunca levanta excepção; forma do resultado da simulação inalterada) e `tests/test_dashboard_montecarlo.py` (o dashboard já não corre uma segunda simulação; os valores mostrados/usados pelo Decision Engine são exatamente os de `analysis["simulation"]`).

---

## 5. Goal Engine

Ficheiro principal: `src/live/engine.py::LiveGoalEngine`.

Não existem fórmulas fechadas de P(golo), P(over), P(BTTS), P(próximo golo) num sentido de probabilidade calibrada — existem **três mecanismos distintos e não integrados** que respondem a partes desta pergunta:

### 5.1 P(golo nos próximos 15 min) — heurística aditiva, `predict_next_goal_probability()`

```
pressure = calculate_pressure(match)          # ver 5.1.1
dominance = calculate_dominance_index(match)   # ver 5.1.2
live_xg_10m = estimate_live_xg(match)          # ver 5.1.3

base_xg = (home_xg_last5 + away_conceded_xg_last5) / 2
score_diff = |home_score − away_score|
draw_bonus = 15 se score_diff==0, 8 se ==1, senão 0
last_goal_bonus = max(0, 15 − (minute − last_goal_minute))   # se houver golo recente
time_factor = min(minute/90, 1)

pressure_score = pressure + draw_bonus + last_goal_bonus + min(base_xg×10, 20) + red_cards×5
pressure_score = clamp(pressure_score, 0, 100)

P(golo_15m) = pressure_score×0.60 + dominance×0.15 + min(live_xg_10m×25, 15) + time_factor×10
```
(`live/engine.py:40-72`)

**Isto não é uma probabilidade no sentido estatístico**: é uma soma ponderada de scores heterogéneos (0-100), alguns já clampados, com pesos (0.60, 0.15, 25→cap 15, 10) escolhidos sem calibração documentada, e o resultado é apresentado como percentagem (`next_goal_probability`) e comparado diretamente com odds implícitas em `evaluate_live_market()` (§8). Não há garantia teórica de que os pesos somem para produzir uma distribuição bem calibrada — de facto, o próprio range de saída não está matematicamente garantido a ficar em [0,100] pelos termos somados (só está protegido porque `pressure_score` já vem clampado e os outros três termos, na pior das hipóteses, somam no máximo 15+15+10=40, dando um teto teórico de 60+15+... na prática fica normalmente dentro do intervalo mas por construção informal, não por prova).

#### 5.1.1 Pressure (`calculate_pressure`, `live/engine.py:8-17`)
```
raw = dangerous_attacks_10m×1.2 + shots_10m×2.5 + shots_on_target_10m×4.0 + corners_10m×1.5
pressure = clamp(raw×0.7 + previous_pressure×0.3, 0, 100)
```
Suavização exponencial simples (EWMA de fator 0.7/0.3) sobre um índice aditivo de volume de jogo. Pesos arbitrários, sem base empírica documentada no código.

#### 5.1.2 Dominance Index (`calculate_dominance_index`, `live/engine.py:19-29`)
```
dominance = possession×0.3 + min(dangerous_attacks_10m/15×35, 35) + min(shots_10m/8×35, 35)
```

#### 5.1.3 xG estimado ao vivo (`estimate_live_xg`, `live/engine.py:31-38`)
```
xG_10m = (shots_10m − shots_on_target_10m)×0.08 + shots_on_target_10m×0.32 + corners_10m×0.05
```
Aproximação de xG por regra fixa por remate (0.08 para remates fora do alvo, 0.32 para remates ao alvo, 0.05 por canto) — não é um modelo de xG treinado (não usa localização do remate, ângulo, tipo de jogada, etc.), é uma constante por tipo de evento.

Existe uma **segunda implementação paralela e ligeiramente diferente** de pressão em `src/live/pressure.py::PressureEngine.score()` (usada por `live_monitor.py`, não por `live/engine.py`):
```
pressure = min(minute,90)×0.6
         + (20 se diff==0, 15 se diff==1, senão 5)
         + min(max(minute − last_goal_minute, 0), 35)
         + max(0, (2.0 − odds_over)×30)        # só se odds_over fornecido
         + home_xg×6 + away_xga×5 + red_cards×5
```
clamp a 100. **Nem os pesos, nem os termos, nem a escala coincidem** com `LiveGoalEngine.calculate_pressure()` — são dois modelos de "pressão" com o mesmo nome, usados por caminhos diferentes do sistema (`live_monitor.py` para logging/treino; `live/engine.py` para o dashboard), que produzem números diferentes para o mesmo jogo.

Existe ainda uma **terceira e quarta variante** em `src/live/features/pressure.py::PressureIndex` e `src/live/features/attack_score.py::AttackScore` — módulos de features que parecem ser uma tentativa de refatoração/unificação nunca concluída (não são importados por nenhum caminho de produção identificado).

### 5.2 P(over 1.5) / P(over 2.5) / P(BTTS) — via Monte Carlo (§4)

```
P(over_1.5) = média{ 1[total_simulado > 1.5] } sobre N simulações
P(over_2.5) = média{ 1[total_simulado > 2.5] }
P(BTTS)     = média{ 1[home_final>0 AND away_final>0] }
```
(`simulation.py:43-45`) — frequências empíricas sobre as simulações Monte Carlo, matematicamente corretas *dado* os λ de entrada. Como discutido em §4, os λ de entrada no caminho de produção são constantes fixas, não calibradas ao jogo.

### 5.3 P(próximo golo) — não existe uma fórmula de "quem marca a seguir" (P(home marca antes de away))

Apesar do nome da métrica na secção pedida pela auditoria, **não há nenhum cálculo de probabilidade de "próximo golo" no sentido de qual equipa marca primeiro**. O que existe chama-se "next_goal_probability" mas é, na realidade, P(ocorre pelo menos um golo nos próximos 15 minutos, de qualquer equipa) — ver §5.1. Não há decomposição por equipa.

### 5.4 P(golo numa janela X de minutos) — `GoalWindowPredictor` (`src/live/features/goal_window.py`)

Não calcula uma probabilidade — devolve uma **janela categórica** e um nível de confiança fixo por categoria (não calculado):

```
se minute ≥ 85:                    janela "85-90+", confiança fixa 60.0%
senão se pressure_index ≥ 60:      janela [minute+2, minute+10], confiança fixa 85.0%
senão se pressure_index ≥ 40:      janela [minute+5, minute+15], confiança fixa 65.0%
senão:                             "Sem Janela Detetada", confiança fixa 20.0%
```
(`goal_window.py:11-27`). Os valores 60.0/85.0/65.0/20.0 são **constantes fixas por categoria**, não uma probabilidade calculada a partir de dados — duas situações que caem na mesma categoria (ex. pressure_index=61 vs. pressure_index=99) recebem exatamente a mesma "confiança".

---

## 6. Expected Value (EV)

### 6.1 Fórmula dominante (correta): `EV = p·odd − 1`

Confirmada em:
- `src/engine/edge.py::calculate_edge/calculate_ev` (linhas 30, 63): `ev = (prob_model * odd_house) - 1.0`
- `src/engine/value.py:21,27,33`: `"ev": (p_x * odd_x) - 1.0`
- `src/engine/decision.py::DecisionEngine.evaluate_bet` (implícito via full Kelly)
- `src/engine/analyzer.py:8`
- `src/engine/bet_engine.py:77`

Esta é a fórmula padrão e matematicamente correta de EV por unidade apostada (odd decimal, stake=1): `EV = p·(odd−1) − (1−p) = p·odd − 1`.

### 6.2 Edge — duas convenções diferentes coexistem

Existem **duas definições de "edge" com escalas diferentes** ativas ao mesmo tempo:

**Convenção A — edge = EV** (`src/engine/edge.py::edge()` / `calculate_edge()`, linhas 16-39): edge é literalmente o mesmo número que EV (`(p×odd)−1`), devolvido em fração decimal. Usado por `analyzer.py`, `full_engine.py`.

**Convenção B — edge = diferença de probabilidades** (`src/engine/decision.py::DecisionEngine.evaluate_bet`, linha 33): `edge = (p − implied_prob) × 100`, em pontos percentuais. Também usada em `src/engine/live_decision.py:21`: `edge = probability_pct − implied_pct`.

Estas duas quantidades **não são a mesma coisa nem têm a mesma escala** (uma é `p·odd − 1`, fração; a outra é `p − 1/odd`, em pontos percentuais) — e ambas são chamadas "edge" em módulos diferentes do mesmo sistema, comparadas depois contra limiares fixos (`min_edge`) que só fazem sentido para uma das duas convenções. Isto é uma fonte real de confusão/erro se qualquer código futuro (ou o próprio dashboard) misturar valores vindos de módulos diferentes assumindo que "edge" quer sempre dizer a mesma coisa.

### 6.3 Bug confirmado: `src/engine/market.py::analyze_market()`

```python
# src/engine/market.py:16-32
market_probability = implied_probability(odd)      # 0.0–1.0
edge = calculate_edge(model_probability, market_probability)   # ← BUG
ev = calculate_ev(model_probability, odd)
```

`calculate_edge(prob_model, odd_house)` (assinatura em `edge.py:16`) espera uma **odd decimal (>1)** como segundo argumento. Aqui recebe `market_probability`, que é uma **probabilidade (0–1)**. Como a validação interna de `calculate_edge` é `if odd_house <= 1.0: return -1.0` (`edge.py:27`), e `market_probability` é quase sempre `<1.0` (só seria `>1` em odds `<1.0`, que não existem em mercados normais), **`analyze_market()` devolve sistematicamente `edge=-1.0` para praticamente qualquer input**, exceto em casos extremos. Isto não foi corrigido (conforme pedido explícito de não alterar código), mas fica aqui documentado como um bug matemático grave: o único caminho de produção que usa `analyze_market()` seria qualquer código que a importe — não foi encontrado nenhum, pelo que este bug está atualmente dormente (código morto), mas é um risco se algum caminho futuro voltar a usar este módulo sem o corrigir primeiro.

### 6.4 Probabilidade implícita

```
implied_probability(odd) = 1/odd     [edge.py:6-13, decision.py:30, live_decision.py:19]
```
Correto — mas note-se que é a probabilidade implícita **com margem da casa incluída** (a odd de mercado já contém o overround/vig do bookmaker). O sistema nunca remove o overround (não há normalização de um conjunto de odds 1X2 para somarem 100%) antes de calcular edge — o que significa que o "edge" calculado contra qualquer mercado individual já está a comparar a probabilidade do modelo contra uma probabilidade de mercado inflacionada pela margem, tornando a barra para "ter edge" mais alta do que seria contra a probabilidade "justa" do mercado. Isto é conservador (reduz falsos positivos de valor), mas não está documentado como decisão de design explícita em lado nenhum do código.

### 6.5 Simulação de condições de mercado (`src/backtest/market.py`)

`apply_market_conditions(raw_odd, margin=0.05, slippage=0.02)` — aplica margem (5%) e slippage (2%) a uma odd "justa" para simular execução realista em backtesting. Matematicamente:
```
fair_prob = 1/raw_odd
implied_prob_com_margem = fair_prob / (1 - margin)
odd_bookmaker = 1/implied_prob_com_margem
odd_executável = odd_bookmaker × (1 - slippage)
```
Correto como simulação de fricção de mercado, mas os valores `margin=0.05` e `slippage=0.02` são constantes fixas assumidas, não calibradas contra dados reais de execução.

---

## 7. Kelly Criterion

Três implementações distintas, com pequenas diferenças de proteção:

### 7.1 `src/engine/kelly.py::kelly_fraction` / `fractional_kelly`
```
b = odd − 1; q = 1 − p
kelly = (b·p − q) / b
fractional = kelly × fraction   (fraction=0.25 por omissão)
```
Fórmula de Kelly padrão, correta. **Sem cap máximo de stake** — se `p` for muito alto, `fractional_kelly` pode devolver uma fração de banca arbitrariamente grande (ex. `p=0.95, odd=5.0` → Kelly completo ≈0.9375, fracionário a 1/4 ≈0.234, ou seja >23% da banca numa única aposta). Não há proteção contra overestimation do modelo.

### 7.2 `src/engine/dixon_coles.py::calculate_fractional_kelly`
```
kelly_full = (b·p − q) / b
kelly_fractional = kelly_full × fraction     (fraction=0.25)
final_stake = min(kelly_fractional, max_stake_pct)    (max_stake_pct=0.02, i.e. 2% hard cap)
```
Igual à anterior mas **com cap explícito de 2% da banca**. Esta é a versão mais protegida contra volatilidade das três, mas está no módulo órfão (§3.2 — nunca chamado em produção).

### 7.3 `src/engine/decision.py::DecisionEngine.evaluate_bet` (inline, é a que corre no dashboard)
```
full_kelly = (b·p − q) / b
suggested_stake = min(full_kelly × max_kelly_fraction × 100, 5.0)   # cap 5% da banca
```
`max_kelly_fraction=0.25` por omissão (`decision.py:13`). Cap de **5%**, mais permissivo que o cap de 2% de `dixon_coles.py`. É esta versão (cap 5%) que efetivamente corre no `main.py live` (via `dashboard.py:48-52`).

### 7.4 `src/engine/stake.py::calculate_stake` — não é Kelly

Usado por `main.py predict`. É uma **tabela de decisão heurística**, não Kelly:
```
se confidence=="HIGH":   stake = edge×0.5  (se h2h≥5)  ou  edge×0.25 (se h2h<5)
se confidence=="MEDIUM": stake = edge×0.25 (se h2h≥3)  ou  edge×0.15 (se h2h<3)
senão:                   stake = edge×0.05
cap: stake = min(stake, 5); floor: stake = max(stake, 0)
```
(`stake.py:1-44`). Note que `edge` aqui vem de `calculate_edge()` que, na convenção A (§6.2), já é o EV em fração (ex. 0.08), não uma percentagem — multiplicar isto por 0.5 ou 0.25 dá números muito pequenos (0.04, 0.02) comparados ao cap de "5" na linha 33-34 (`if stake > 5: stake = 5`), o que sugere uma **inconsistência de escala**: o cap `>5` parece assumir que `stake` está em percentagem (ex. 5%), mas o valor de entrada (`edge`) está tipicamente entre -1 e 1 (fração). Isto significa que o cap de "5" praticamente nunca é atingido pelos valores reais de edge produzidos por `calculate_edge()`, tornando a proteção de cap **matematicamente inoperante** na prática (a menos que `edge` seja alimentado, nalgum caminho, já em pontos percentuais — o que é precisamente a ambiguidade da convenção dupla descrita em §6.2).

### 7.5 Resumo — proteção contra volatilidade

| Implementação | Fractional Kelly | Cap máximo | Usada em produção? |
|---|---|---|---|
| `kelly.py` | Sim (1/4 por omissão) | **Não** | Indireto (`hybrid_engine.py`, `bet_engine.py`) |
| `dixon_coles.py::calculate_fractional_kelly` | Sim (1/4) | 2% | **Não** (código órfão) |
| `decision.py::DecisionEngine` | Sim (1/4) | 5% | **Sim** (`main.py live`) |
| `stake.py::calculate_stake` | N/A (não é Kelly) | Nominalmente 5, mas inoperante por mismatch de escala | **Sim** (`main.py predict`) |

Não há, em nenhuma das implementações, ajuste de Kelly pela incerteza do modelo (ex. Kelly ajustado por variância da estimativa de p, como em Kelly-Bayesiano ou "Kelly com shrinkage"), apesar de `confidence.py` calcular um `model_std` que poderia alimentar isso — atualmente o `model_std` só entra no cálculo de `confidence` (§8), não no de stake.

---

## 8. Decision Engine

Não existe um único "Decision Engine" — pelo menos **quatro caminhos de decisão diferentes**, com limiares diferentes:

### 8.1 `src/engine/decision.py::DecisionEngine.evaluate_bet` (usado em `main.py live`)

```
implied_prob = 1/odd
edge = (p − implied_prob) × 100                    # convenção B, pontos percentuais
full_kelly = (b·p − q)/b
se full_kelly > 0 E edge ≥ min_edge (5.0):
    stake = min(full_kelly × 0.25 × 100, 5.0)
    action = "BET"
senão:
    stake = 0; action = "PASS"
```
Critério: **edge ≥ 5 p.p. E Kelly positivo** → BET. Não há estado intermédio "WAIT" neste caminho — é binário BET/PASS.

### 8.2 `src/engine/decision.py::evaluate_decision` (função module-level, usada por `hybrid_engine.py`, `predict.py` via `make_decision`)

```
se edge ≥ 5 E ev > 0:  "BET"
senão se edge > 0:      "WAIT"
senão:                  "PASS"
```
Aqui `edge` está implicitamente na convenção A (EV-like, `p×odd−1`) vindo de `calculate_edge()`, mas o limiar `≥5` sugere pontos percentuais (convenção B) — o mesmo problema de mismatch de escala do §6.2/§7.4 repete-se aqui: se `edge` vier de `calculate_edge()` (tipicamente entre -1 e +0.3), a condição `edge ≥ 5` **nunca é verdadeira**, e o sistema cairia sempre em "WAIT" ou "PASS", nunca em "BET", por este caminho — a menos que a variável `edge` que chega aqui já esteja em pontos percentuais (o que depende de qual função a montante gerou o valor). Esta ambiguidade não pode ser resolvida por leitura estática sem rastrear, chamada a chamada, qual `edge` (convenção A ou B) chega a cada `evaluate_decision`.

**Nota crítica de bug adicional**: `decision.py` define `make_decision` **duas vezes** — uma em `decision.py:58-75` (com um "WAIT" intermédio ligeiramente diferente) e outra em `decision.py:93-98` (delega para `evaluate_decision`). Em Python, a segunda definição **sobrepõe-se silenciosamente à primeira** — qualquer importação de `make_decision` (usada em `cli/predict.py:9`) obtém sempre a segunda versão. A primeira (linhas 58-75) é código morto inacessível, mas continua no ficheiro como se fosse ativo, o que é enganador para quem audita ou modifica o código.

### 8.3 `src/engine/analyzer.py::analyze_bet` (usado por `hybrid_engine.py`, `bet_engine.py`, `ranking.py`)

```
se bet_edge ≥ 5 E ev ≥ 0.08:  "VALUE BET"
senão se bet_edge ≥ 3:          "WATCH"
senão:                          "PASS"
```
Aqui `bet_edge` vem de `edge()` (convenção A, fração tipo EV) e o limiar é `≥5` — de novo o mesmo mismatch: se `bet_edge` estiver tipicamente entre -1 e 0.3, `bet_edge ≥ 5` é (quase) sempre falso, e `bet_edge ≥ 3` também. Isto significaria, na prática, que `analyze_bet` produziria quase sempre "PASS" para inputs realistas, exceto em odds/probabilidades extremas. **Este é o terceiro local independente onde a mesma classe de bug de escala (edge em fração vs. edge em pontos percentuais) aparece.**

### 8.4 `src/engine/live_decision.py::evaluate_live_market` (usado no live dashboard, mercado "NEXT GOAL")

```
implied = (1/odd) × 100
edge = probability_pct − implied           # em pontos percentuais, consistente aqui
se edge ≥ 10:  "BET VALUE"
se edge ≥ 3:   "WATCH"
senão:          "PASS"
```
Este é internamente consistente (ambos os operandos em pontos percentuais), ao contrário de §8.2/§8.3.

### 8.5 `src/engine/filter.py::is_valid_bet` (usado por `ranking.py::create_ranking`)

Critérios adicionais de corte, aplicados sobre o dicionário `result` já calculado a montante:
```
decision == "VALUE BET"  E  edge ≥ 5  E  ev ≥ 10  E  confidence != "LOW"  E  odd ≥ 1.70
```
Note que aqui `ev ≥ 10` sugere que `ev` está em **percentagem** (ex. 10 = 10%), consistente com `analyze_bet()` que devolve `"ev": round(ev*100, 2)` (`analyzer.py:22`) — ou seja, este filtro está calibrado para a saída de `analyze_bet`, não para a saída "crua" de `calculate_ev()` (fração). Esta é a única cadeia de decisão onde a conversão de escala parece ter sido feita corretamente ponta-a-ponta.

### 8.6 Resumo — critérios de BET por caminho

| Caminho | Limiar de edge | Limiar de EV | Confiança usada? | Odds mínima? |
|---|---|---|---|---|
| `DecisionEngine.evaluate_bet` (live) | ≥5 p.p. | — (via Kelly>0) | Não | Não |
| `evaluate_decision` (predict/hybrid) | ≥5 (escala ambígua) | >0 | Não | Não |
| `analyze_bet` | ≥5 (escala ambígua) | ≥0.08 | Não | Não |
| `evaluate_live_market` | ≥10 p.p. | — | Não | Não |
| `is_valid_bet` (filtro final) | ≥5 (escala %) | ≥10 (escala %) | Sim (exclui LOW) | ≥1.70 |

Não há, em nenhum caminho, um limiar dependente da **confiança do modelo** ajustando o limiar de edge exigido (ex. exigir edge maior quando a confiança é mais baixa) — a confiança (`confidence.py`) é calculada mas só é usada como filtro binário em `is_valid_bet` (exclui "LOW"), nunca modula o limiar de decisão de forma contínua.

---

## 9. Machine Learning — inventário completo

### 9.1 XGBoost + CalibratedClassifierCV (sintético) — **em produção**

- **Ficheiro de treino:** `src/model/train.py`
- **Algoritmo:** `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05)` envolto em `CalibratedClassifierCV(cv=3, method="sigmoid")`.
- **Features (10):** minute, dangerous_attacks_10m, shots_on_target_10m, shots_10m, corners_10m, possession, previous_pressure, shots_ratio (derivada), danger_intensity (derivada), xg_diff.
- **Target:** golo nos próximos 10 min (binário), gerado sinteticamente por `1[uniform(0,1) < sigmoid(score_signal − 2.0)]`, onde `score_signal` é ele próprio uma combinação linear sintética das mesmas features sintéticas — ou seja, o "sinal" que o modelo aprende é uma relação que os próprios autores do script inventaram, não uma relação extraída de futebol real.
- **Dataset:** 5000 amostras 100% sintéticas (`np.random`, seed=42). Zero jogos reais.
- **Treino:** todo o dataset sintético, sem hold-out reportado (a `CalibratedClassifierCV(cv=3)` faz cross-val interna só para a calibração, não há um teste independente reportado com métricas).
- **Validação:** nenhuma métrica é impressa ou guardada (`train_and_save_model` não calcula AUC/Brier/log-loss em nenhum ponto).
- **Onde é usado:** `src/model/ml_predictor.py::LiveMLPredictor`, carregado por `src/report/dashboard.py:33` (linha "🤖 ML Goal" no output do `main.py live`).

### 9.2 Pipeline de treino rigoroso — **não está em produção**

- **Ficheiro:** `src/training/train_model.py`.
- **Algoritmos candidatos (6):** RandomForest, ExtraTrees, BalancedRandomForest (imbalanced-learn), GradientBoosting, LightGBM, HistGradientBoosting — todos com `max_depth=4`, `class_weight="balanced"` onde aplicável.
- **Features (11):** current_minute, home_score, away_score, dangerous_attacks_10m, shots_on_target_10m, corners_10m, live_odd_over, pressure, live_xg, red_cards, possession — extraídas diretamente das colunas de `data/live_history.db`.
- **Target:** `goal_in_next_15m` (booleano) — dados reais logados pelo `live_monitor.py` em produção.
- **Dataset:** `data/training_dataset.csv`, gerado por `src/training/build_dataset.py` a partir de `data/live_history.db` — **1535 linhas, 75 jogos distintos**, distribuição de classes 1036 negativos / 499 positivos (≈32.5% positivos) — confirmado por consulta direta à base de dados nesta auditoria.
- **Validação:** `StratifiedGroupKFold`/`GroupKFold` com 5 folds, **agrupado por `match_id`** (garante que snapshots do mesmo jogo nunca aparecem simultaneamente em treino e teste — proteção correta contra leakage temporal/dentro-do-jogo) — com um `assert` explícito de "zero overlap" por fold (`train_model.py:350-353`).
- **Métricas:** accuracy, precision, recall, F1, ROC-AUC por fold e agregadas (média ± desvio-padrão).
- **Calibração:** `CalibratedClassifierCV(method="sigmoid", cv=<folds da CV>, ensemble=True)` sobre o modelo vencedor, com comparação explícita de Brier score e log-loss antes/depois, incluindo uma verificação de que a amostra out-of-fold "antes" e "depois" é a mesma população (proteção contra comparações inválidas).
- **Otimização de threshold:** grid search em `[0.05, 0.95]` passo 0.01, otimizando F1 sobre previsões out-of-fold (nunca sobre o modelo final treinado com 100% dos dados — proteção correta contra otimismo de threshold).
- **Ficheiro `src/training/audit_dataset.py`:** módulo adicional e independente de deteção de data leakage — treina um RandomForest **por feature isolada** e assinala qualquer feature cujo AUC sozinha exceda 0.90 como possível leakage. Este é o único módulo de todo o repositório que audita ativamente o próprio dataset de treino antes de confiar nele.
- **Output:** `models/live_goal_model.pkl` + `models/live_goal_model_metrics.json` + `models/train_validation_report.md` + `models/calibration_report.md` + `models/threshold_optimization_report.md` + `models/dataset_leakage_report.{json,md}`.
- **Onde é usado:** `src/live/ml_predictor.py::MLGoalPredictor` carrega `models/live_goal_model.pkl` — **mas este ficheiro não existe no repositório** (a pasta `models/` nem sequer existe — só `models_data/` existe, com o modelo sintético do §9.1). `MLGoalPredictor` não é importado por nenhum outro módulo do projeto (confirmado por pesquisa global). **Este pipeline de treino é, no estado atual do repositório, trabalho de engenharia estatística válido e bem construído que está completamente desligado da produção.**

### 9.3 RandomForest de remates (mercado de shots, não de golos) — **em produção via workflow diário, mas para o mercado errado**

- **Ficheiros:** `src/engine/predict_today.py`, `research/pressure_shots/predict_engine_bridge.py`, `research/pressure_shots/models/random_forest.py`, `random_forest_v2.py`.
- **Algoritmo:** `RandomForestClassifier(n_estimators=100, max_depth=5)` (classificação over/under) ou `RandomForestRegressor(n_estimators=200-300)` (regressão do total de remates).
- **Features:** `attack_avg_last5`, `dangerous_attack_avg_last5`, `ball_safe_avg_last5`, `total_shots_avg_last5`, `shots_on_target_avg_last5`, e diferenças contra o adversário (`*_difference`) — rolling averages de 5 jogos calculadas em `research/pressure_shots/features/build_history_features.py` com `shift(1).rolling(5, min_periods=1).mean()` (correto: usa `shift(1)` para não incluir o próprio jogo no rolling — sem leakage temporal óbvio neste ponto).
- **Target:** `total_shots > 12.5` (classificação) ou `total_shots` (regressão) — mercado de **remates totais**, não golos, não 1X2.
- **Dataset:** `research/pressure_shots/features_v2.csv` (dados de jogos reais processados) — com fallback para dados 100% sintéticos (`generate_synthetic_historical_data`) quando o CSV não existe.
- **Treino/validação:** split 80/20 posicional (não é `train_test_split` aleatório em `random_forest_v2.py`/`predict_shots.py`— é um corte pelas primeiras 80% linhas, o que só é temporalmente correto se o CSV já vier ordenado cronologicamente; isto não é garantido explicitamente no código de leitura). `predict_engine_bridge.py` já usa `train_test_split(test_size=0.2, random_state=42)`, aleatório.
- **Métricas:** MAE, RMSE (regressão); nenhuma métrica de classificação reportada para a variante classificadora usada em produção (`predict_today.py`).
- **Mismatch de mercado (achado grave):** em `predict_today.py:151-159` e `predict_engine_bridge.py:72-84`, a probabilidade de saída deste modelo (P(remates>12.5)) é passada para `run_pipeline(prob_model=prob, odd_house=odd, ...)` onde `odd` vem de `odds/?event_id=` — **a odd do mercado 1X2 do evento** (`predict_today.py:71-73`, comentário `decimal_odds` sem especificar mercado, mas o fallback `2.00` e o contexto de "BOLETIM DE APOSTAS" sugerem odds de resultado, não de remates). Ou seja: **o sistema compara a probabilidade de um mercado (remates) com a odd de outro mercado (resultado/1X2)**, o que invalida completamente o cálculo de EV/edge para este caminho — não são o mesmo evento de aposta.

### 9.4 Resumo ML

| Modelo | Dataset real? | Validação rigorosa? | Mercado correto? | Em produção? |
|---|---|---|---|---|
| XGBoost calibrado (`model/train.py`) | **Não (sintético)** | Não | N/A (golos) | **Sim** |
| Pipeline `training/train_model.py` | **Sim** (1535 snapshots, 75 jogos) | **Sim** (GroupKFold, leakage check, calibração) | N/A (golos) | **Não** (artefacto ausente) |
| RandomForest remates (`predict_today.py`) | Sim (quando CSV existe) | Parcial (sem métricas de classificação) | **Não** (prob. de remates vs. odd de 1X2) | Sim (workflow diário) |
| RandomForest H2H (`model/predictor.py`) | N/A (não é ML, é regras) | — | — | Sim (`main.py predict`) |

---

## 10. Base de Dados — `data/live_history.db`

### 10.1 Estrutura

Tabela única `match_snapshots` (schema em `src/backtest/logger.py::init_db`, linhas 18-43), com colunas adicionadas incrementalmente via `ALTER TABLE ... ADD COLUMN` idempotente (linhas 46-64) — abordagem defensiva razoável para evolução de schema sem migrações formais.

**Estado real (consultado nesta auditoria):**
- **1535 snapshots**, **75 jogos distintos** (`match_id`).
- Distribuição do label `goal_in_next_15m`: **1036 negativos, 499 positivos** (≈32.5% de taxa base positiva).
- Média de ~20 snapshots por jogo (1535/75), consistente com o `live_monitor.py` correr periodicamente durante o jogo (uma vez por invocação do workflow `live_logger.yml`, que é `workflow_dispatch` — acionado manualmente ou por outra automação externa, não por cron nesta configuração).

### 10.2 Como aprende / como gera labels — três implementações inconsistentes

**(a) `src/backtest/logger.py::update_outcomes`** (linhas 126-144) — corre **dentro** de `live_monitor.py` a cada execução, para cada evento ao vivo:
```sql
UPDATE match_snapshots SET goal_in_next_15m = ?
WHERE match_id = ? AND current_minute BETWEEN (current_minute-18) AND (current_minute-12)
  AND goal_in_next_15m IS NULL
```
Isto assume que o `live_monitor.py` corre a cada ~15 minutos e usa uma janela de tolerância de ±3 min (`[-18,-12]`) à volta do alvo de "15 minutos atrás", para apanhar snapshots que caiam perto disso independentemente do timing exato de execução do workflow. Só atualiza linhas onde o label ainda é `NULL` (não sobrescreve).

**(b) `src/training/create_labels.py`** — corre a seguir, no mesmo workflow (`live_logger.yml`, passo "4.5"), sobre **toda a tabela**, sobrescrevendo incondicionalmente (não verifica `IS NULL`):
```sql
UPDATE match_snapshots SET goal_in_next_15m = CASE WHEN EXISTS (
    SELECT 1 FROM match_snapshots b
    WHERE b.match_id = match_snapshots.match_id
      AND b.home_score + b.away_score > match_snapshots.home_score + match_snapshots.away_score
      AND b.current_minute > match_snapshots.current_minute
      AND b.current_minute <= match_snapshots.current_minute + 15
) THEN 1 ELSE 0 END
WHERE current_minute IS NOT NULL
```
Esta versão usa `current_minute` diretamente (não `timestamp`) e uma janela `(minute, minute+15]` — **estruturalmente diferente** da janela `[-18,-12]` de (a), e corre por **último** no workflow, pelo que **o label final gravado na base de dados é sempre determinado por (c)/(b), não por (a)**, apesar de (a) correr primeiro. Isto significa que o trabalho de (a) (que usa timestamp de execução real, mais robusto a gaps de execução do workflow) é **sempre substituído** pelo recalculo de (b), que assume implicitamente que existe sempre um snapshot dentro de `(minute, minute+15]` para qualquer jogo — o que só é garantido se `live_monitor.py` tiver corrido com cadência suficientemente densa dentro desse jogo.

**(c) `src/backtest/labeler.py`** — uma **terceira versão**, baseada em `timestamp` (não `current_minute`):
```sql
UPDATE match_snapshots SET goal_in_next_15m = 1 WHERE id IN (
  SELECT a.id FROM match_snapshots a JOIN match_snapshots b
  ON a.match_id=b.match_id AND b.timestamp > a.timestamp
     AND b.timestamp <= datetime(a.timestamp, '+15 minutes')
  WHERE b.home_score != a.home_score OR b.away_score != a.away_score
)
```
Nunca atribui `0` explicitamente (só faz `UPDATE ... SET = 1` condicional; linhas que não correspondem ficam com o valor anterior, que pode ser `NULL`). **Não é chamado por nenhum workflow nem por nenhum outro módulo** (confirmado por pesquisa global de imports) — é um script standalone, órfão, cuja lógica diverge das outras duas.

### 10.3 Como melhora as previsões

Formalmente, o caminho para a base de dados influenciar previsões futuras é: `live_history.db` → `build_dataset.py` → `training_dataset.csv` → `train_model.py` → `models/live_goal_model.pkl` → `MLGoalPredictor`. Como documentado em §9.2, **este caminho nunca chega a produção** no estado atual do repositório — o modelo carregado em produção (`models_data/xgboost_live_v1.pkl`) não deriva, direta ou indiretamente, de nenhum dado gravado em `live_history.db`. **A base de dados, apesar de ativamente alimentada por um workflow automático diário/manual, não fecha o ciclo de aprendizagem em produção.** É consumida apenas por análises manuais/scripts de auditoria (`audit_dataset.py`) e pelo pipeline de treino desligado.

---

## 11. Validação Estatística — por componente

| Componente | Solidez matemática | Simplificação? | Adequado para produção? | Confiança |
|---|---|---|---|---|
| Dixon-Coles/Poisson (`dixon_coles.py`) | Fórmula correta e fiel à literatura | λ não é estimado dos dados (falta metade do método clássico); nunca chamado em produção | Não (código órfão) | **Baixo** (não por a matemática estar errada, mas por estar desligado do sistema) |
| Monte Carlo (`simulation.py`) | Mecânica correta (Poisson + agregação de frequências) | λ de entrada são constantes fixas em produção; ignora placar real num dos dois caminhos | Não, no estado atual (input inválido) | **Baixo** |
| Goal Engine heurístico (`live/engine.py`) | Sem base estatística — soma ponderada arbitrária | Pesos não calibrados; múltiplas implementações de "pressão" divergentes | Não | **Baixo** |
| Modelo H2H (`model/predictor.py`) | Heurística simples, sem calibração | Clamp arbitrário [35,65]; ignora quase toda a informação disponível (xG, forma, odds) | Marginal — funciona como filtro grosseiro, não como estimador de probabilidade | **Baixo-Médio** |
| XGBoost sintético (produção) | Algoritmo correto, calibração formalmente aplicada | Treinado 100% em dados sintéticos — não generaliza para futebol real | **Não** | **Baixo** |
| Pipeline `train_model.py` (desligado) | GroupKFold, leakage check, calibração, threshold ótimo — boas práticas | Dataset pequeno (75 jogos, 499 positivos) limita generalização mesmo sendo bem validado | Sim, *se* ligado à produção e com mais dados | **Médio** (limitado pelo tamanho da amostra, não pelo método) |
| RandomForest remates | Rolling features corretas, mas comparado com odd de mercado errado | Mismatch de mercado (§9.3) invalida o EV calculado | Não, no caminho atual | **Baixo** |
| EV (`edge.py`) | Fórmula `p·odd−1` correta | Não remove overround; não afeta corretude da fórmula em si | Sim, isoladamente | **Alto** (a fórmula em si) |
| Edge — convenção dupla | Duas definições coexistentes, ambíguas entre módulos | Risco de comparação de grandezas incompatíveis | Não, sem normalização de convenção | **Baixo** |
| Kelly (`kelly.py`, `dixon_coles.py`, `decision.py`) | Fórmula padrão correta em todos os locais | Caps de proteção inconsistentes (nenhum / 2% / 5%); nenhum ajuste por incerteza do modelo | Parcial — depende de qual das 3 versões é usada | **Médio** (a versão com cap 5% em produção é razoável isoladamente) |
| Decision Engine | Lógica de threshold simples e correta *isoladamente* em cada módulo | Limiares aplicados a "edge" com escalas diferentes consoante o caminho (§8) | Não, de forma agregada | **Baixo** |
| Base de dados / labels | SQL correto em cada implementação individual | 3 implementações divergentes da mesma label; a que "vence" não é a mais robusta (usa minute, não timestamp) | Parcial | **Médio-Baixo** |

---

## 12. Benchmark — comparação com boas práticas de modelos quantitativos em apostas desportivas

### 12.1 Pontos fortes (relativos a um sistema de research em fase inicial)

- **O núcleo Dixon-Coles está corretamente implementado** e é o mesmo modelo usado por várias casas quantitativas e projetos académicos de referência para mercados de golos — a fórmula de τ e a normalização estão certas.
- **O pipeline `training/train_model.py` segue boas práticas reais de ML para dados desportivos**: agrupamento por `match_id` na validação cruzada (evita leakage entre snapshots do mesmo jogo — um erro extremamente comum em projetos amadores de apostas ao vivo), calibração explícita com comparação antes/depois, e otimização de threshold sobre previsões out-of-fold (nunca sobre o modelo final). Isto está ao nível do que se espera de um pipeline de produção sério — é a peça de maior qualidade de todo o repositório.
- **`audit_dataset.py`** — a existência de um script dedicado a detetar data leakage por feature isolada (AUC>0.90 como sinal de alarme) é uma prática de maturidade acima da média para este tipo de projeto.
- **Backtesting com fricção de mercado** (`backtest/market.py`, `research/backtest_engine.py`) — simular margem e slippage, e reportar métricas separadas "ideal vs. stressed", é uma prática correta que muitos projetos amadores ignoram.
- **Fractional Kelly com cap explícito** (na versão `dixon_coles.py`) é a prática recomendada pela literatura de gestão de banca (Kelly completo é conhecidamente demasiado agressivo sob incerteza de estimação de `p`).
- **Documentação de filosofia de projeto clara** (`docs/01_project_scope.md`, `docs/02_architecture.md`): "não prevemos jogos, identificamos ineficiências de preço" e "toda a feature deve ter hipótese futebolística + definição matemática + validação histórica" são princípios corretos e alinhados com boas práticas de trading quantitativo. **O código atual não cumpre ainda estes princípios que o próprio projeto define para si mesmo** — nenhuma das heurísticas do Goal Engine ou do modelo H2H tem uma "validação histórica" documentada, apesar do princípio de design explicitamente exigir isso.

### 12.2 Limitações face a boas práticas de modelos quantitativos

- **Ausência de estimação de parâmetros a partir de dados históricos para o modelo de golos.** Um sistema quantitativo de referência (ex. implementações públicas de Dixon-Coles, ou os modelos internos descritos na literatura de apostas esportivas) estima ataque/defesa por equipa via máxima verosimilhança sobre uma janela histórica, com time-decay (peso maior a jogos recentes — o próprio repositório já tem uma função `decay.py::apply_exponential_decay` pronta para isto, mas não está ligada ao Dixon-Coles). Aqui, λ chega de fora como argumento avulso ou como constante fixa.
- **Nenhum tracking de Closing Line Value (CLV)** apesar de estar listado como métrica de sucesso em `docs/01_project_scope.md:54-57` e em `docs/02_architecture.md:100`. Não há nenhum módulo que registe a odd no momento da aposta vs. a odd de fecho.
- **Nenhuma calibração formal (reliability diagram / Brier score) do output final apresentado ao utilizador** — só o pipeline desligado (`train_model.py`) calcula Brier score, e apenas para o seu próprio modelo interno, não para o número que acaba no dashboard.
- **Múltiplas fontes de verdade para o mesmo conceito** (edge, pressão, lambda, EV) é o oposto do princípio "uma fonte de verdade" recomendado em sistemas de trading — em produção quantitativa real, isto é uma classe de risco operacional tão importante quanto erros de modelo (um valor "edge" errado por mismatch de unidades pode gerar apostas de tamanho incorreto silenciosamente).
- **Ausência de position sizing agregado / exposição por jogo ou por liga** — cada aposta é dimensionada isoladamente (Kelly por mercado), sem limite de exposição simultânea correlacionada (ex. várias apostas no mesmo jogo ou em jogos correlacionados da mesma liga/dia).
- **Sem seed control na simulação Monte Carlo** — reduz reprodutibilidade de decisões (duas execuções idênticas do mesmo jogo, no mesmo minuto, podem gerar decisões diferentes por ruído de simulação combinado com proximidade a um limiar).
- **Datasets muito pequenos para os modelos que usam dados reais** — 75 jogos / 1535 snapshots é uma amostra pequena para qualquer modelo com múltiplas features (risco de overfitting mesmo com CV bem feita); nenhum dos relatórios gerados (`train_validation_report.md`, etc.) está presente no repositório para confirmar os números reais obtidos, porque o pipeline nunca foi corrido em produção.
- **Sem versionamento de modelo nem de dataset** — não há hash/timestamp associado ao `.pkl` em produção que permita saber, a partir do dashboard, com que dados/versão de código o modelo foi treinado.

### 12.3 Nível de maturidade global (referencial informal, não uma escala normalizada)

- **Camada de matemática de mercado (odds → probabilidade implícita → EV):** madura na fórmula, mas com risco operacional de escala/convenção.
- **Camada de modelação de golos (Poisson/Dixon-Coles):** teoricamente correta, operacionalmente inexistente (não integrada).
- **Camada heurística ao vivo (pressão/dominância/Goal Engine):** protótipo de investigação, sem validação estatística.
- **Camada de ML:** dois extremos — um pipeline de treino de qualidade de produção mas desligado, e um modelo efetivamente em produção treinado sobre dados sintéticos.
- **Camada de dados/labels:** ativamente recolhida, mas com lógica de rotulagem internamente inconsistente.
- **Camada de decisão/gestão de risco:** logicamente razoável por módulo, mas fragmentada em convenções incompatíveis entre módulos.

---

## 13. Recomendações Futuras (apenas registo — nenhuma implementada nesta auditoria)

Por instrução explícita, nenhuma destas recomendações foi implementada. Ficam registadas para decisão e priorização futura pelo dono do projeto:

1. Escolher **um único pipeline de produção** (`live` ou `predict_today`) e desativar/arquivar os restantes, para eliminar a fragmentação em 4+ caminhos paralelos.
2. Unificar a definição de "edge" para uma única convenção (recomenda-se pontos percentuais, `p − 1/odd`, por ser a mais intuitiva para leitura humana) em todos os módulos, e auditar cada limiar (`≥5`, `≥3`, `≥10`) à luz dessa unificação.
3. Ligar `calculate_dynamic_lambda()` (ou uma versão calibrada do Dixon-Coles) efetivamente à simulação Monte Carlo mostrada no dashboard, em vez das constantes fixas atuais.
4. Corrigir o provider `APIMatchProvider.get_live_match` para não forçar a zero as métricas de volume de jogo (ou documentar explicitamente que esse caminho está incompleto/em desenvolvimento).
5. Decidir entre as três implementações de labeling de `goal_in_next_15m` e remover as outras duas, preferindo a mais robusta a gaps de execução (baseada em `timestamp`, não em `current_minute` puro).
6. Treinar `train_model.py` sobre o dataset real acumulado, commitar/publicar `models/live_goal_model.pkl`, e substituir o modelo sintético (`models_data/xgboost_live_v1.pkl`) atualmente em produção.
7. Corrigir o mismatch de mercado em `predict_today.py`/`predict_engine_bridge.py` (probabilidade de remates comparada com odd de 1X2).
8. Implementar tracking de CLV, conforme já prometido na documentação de arquitetura do próprio projeto.
9. Adicionar seed determinístico à simulação Monte Carlo para reprodutibilidade de decisões.
10. Consolidar as quatro implementações de "pressão"/"dominância" (`live/engine.py`, `live/pressure.py`, `live/features/pressure.py`, `live/features/attack_score.py`) numa única.

---

## 14. Definição Oficial de Edge (correção aplicada)

**Data:** 2026-08-03. Âmbito: correção pontual de consistência matemática, seguindo os achados das secções 6.2 e 6.3. Ao contrário do resto deste documento (auditoria estática, sem alterações), esta secção documenta **código efetivamente alterado**.

### 14.1 Definição escolhida

```
edge = prob_model − implied_probability(odd_house)
     = prob_model − (1 / odd_house)
```

implementada como a única função oficial `calculate_edge()` em `src/engine/edge.py`. Recebe `prob_model` em fração (0.0–1.0] e `odd_house` como odd decimal (>1.0); devolve o edge na mesma escala fracionária — para apresentação em pontos percentuais, multiplica-se o resultado por 100.

Esta é a **Convenção B** identificada em §6.2 (`edge = p − implied_prob`), não a Convenção A (`edge = EV = p·odd − 1`, que era a fórmula até agora implementada em `calculate_edge`).

### 14.2 Porque foi escolhida esta convenção e não a outra

1. **É a que o próprio documento já recomendava** (§13, recomendação 2): "unificar a definição de edge para pontos percentuais, `p − 1/odd`, por ser a mais intuitiva para leitura humana". Esta correção não inventa uma fórmula nova — implementa a recomendação que já constava da auditoria.
2. **É a única das duas convenções que já funcionava corretamente nos limiares de decisão existentes.** `DecisionEngine.evaluate_bet` (`min_edge=5.0`), `evaluate_live_market` (limiares 10/3) e `is_valid_bet` (`edge<5`) já assumem edge em pontos percentuais, na gama aproximada de −100 a +100. A Convenção A (edge=EV, fração tipicamente entre −1 e +0.3) tornava esses mesmos limiares matematicamente inoperantes: `analyzer.py::analyze_bet` (`bet_edge >= 5`) e `evaluate_decision` (`edge >= 5`) nunca disparariam "BET"/"VALUE BET" para valores realistas, porque uma fração de EV quase nunca atinge 5. Escolher a Convenção B corrige esse mismatch em vez de o deslocar para outro sítio.
3. **Mantém o Expected Value intacto e distinto do Edge.** `calculate_ev(prob_model, odd_house) = prob_model·odd_house − 1` não foi alterado — continua a ser a fórmula correta e já validada (§6.1, confiança "Alto"). Com a Convenção A, "edge" e "EV" eram literalmente o mesmo número com dois nomes (`expected_value()` chamava a mesma função que `edge()`), o que é a própria origem da confusão de nomenclatura descrita em §6.2. Com a Convenção B, Edge (desvio de probabilidade face ao mercado) e EV (retorno esperado por unidade apostada) passam a ser duas grandezas relacionadas mas distintas, cada uma com a sua própria fórmula e função — tal como os seus nomes sugerem.
4. **É consistente com os fixtures de teste já existentes no repositório.** Os scripts de demonstração `src/tools/test_report.py`, `test_ranking.py`, `test_filter.py`, `test_stake.py` e `test_decision.py` já usavam valores como `edge=7.38, ev=15.5` para `odd=2.10`. Esses valores só emergem exatamente de `prob_model=0.55`: `edge=(0.55−1/2.10)×100=7.38` e `ev=(0.55×2.10−1)×100=15.5`. Ou seja, a Convenção B com `prob_model` em fração já era, implicitamente, o desenho pretendido pelo resto do sistema — só não estava a ser respeitado nos pontos onde o bug existia.

### 14.3 Validação adicionada a `calculate_edge`

`calculate_edge` passou a validar explicitamente os seus argumentos e a lançar `ValueError` para:
- `prob_model` fora de `(0.0, 1.0]` (probabilidade inválida, incluindo o caso do bug histórico: alguém passar a probabilidade em escala 0–100, ex. `55` em vez de `0.55`);
- `odd_house <= 1.0` (odd inválida, incluindo o caso do bug histórico: alguém passar uma probabilidade de mercado, tipicamente <1.0, no lugar da odd).

Antes desta correção, ambos os casos deviam silenciosamente `-1.0`, um valor que parece um edge legítimo (edge de −100%) e por isso mascarava o erro — foi precisamente essa devolução silenciosa que permitiu ao bug de `market.py` (§14.4) passar despercebido. `calculate_ev` mantém deliberadamente o comportamento anterior (sentinela `-1.0`, sem exceção), por não ser o alvo desta correção e por já estar validada como correta (§6.1).

### 14.4 Bug corrigido em `src/engine/market.py`

- **Função afetada:** `analyze_market()`.
- **Porque esperava uma odd:** chama `calculate_edge(prob_model, odd_house)`, cujo segundo argumento é, por contrato, uma odd decimal (>1.0), usada para derivar a probabilidade implícita de mercado internamente.
- **Porque recebia uma probabilidade:** o código chamava `calculate_edge(model_probability, market_probability)`, onde `market_probability = implied_probability(odd)` já é a probabilidade implícita (0.0–1.0) — ou seja, a mesma quantidade que `calculate_edge` deveria calcular internamente estava a ser passada como se fosse a odd.
- **Impacto:** como `market_probability` é (quase) sempre `<= 1.0`, a validação (agora explícita, antes um `return -1.0` silencioso) fazia com que `analyze_market()` devolvesse edge = −1.0 (ou, com a correção, levantasse `ValueError`) para praticamente qualquer combinação real de odds/probabilidade — o cálculo de valor de mercado deste módulo estava sistematicamente inutilizável.
- **Correção:** `calculate_edge(model_probability, odd)` — passa a própria odd, tal como já acontecia (corretamente) na chamada equivalente a `calculate_ev`.
- **Alcance da correção:** foi encontrado exatamente o mesmo padrão de bug, com o mesmo par de argumentos trocados, em `src/cli/predict.py` — o caminho ativo de `main.py predict`. Como o pedido desta tarefa é eliminar inconsistências de Edge (e não apenas corrigir a linha citada em `market.py`), esse segundo local foi corrigido da mesma forma (ver relatório técnico da PR para detalhe completo dos ficheiros alterados).

### 14.5 Módulos que passam a reutilizar `calculate_edge`

`src/engine/decision.py::DecisionEngine.evaluate_bet`, `src/engine/live_decision.py::evaluate_live_market` e `src/engine/bet_engine.py::evaluate_bet` recalculavam a mesma fórmula (`p − 1/odd`) de forma independente. Todos passaram a chamar `calculate_edge()` centralizada, eliminando a duplicação de fórmula identificada em §6.2 sem alterar o valor numérico produzido para inputs válidos.

---

*Fim do relatório de auditoria original (secções 1–13). A secção 14 documenta a correção de consistência de Edge aplicada posteriormente, incluindo os ficheiros de código alterados. A secção 15 documenta a integração do Dixon-Coles na pipeline de produção.*

---

## 15. Dixon-Coles entra em produção

**Data:** 2026-08-03. Âmbito: integrar o modelo Dixon-Coles/Poisson já existente (`src/engine/dixon_coles.py`) na pipeline de produção `main.py predict`. **Nenhuma fórmula do Dixon-Coles, do Monte Carlo, do Kelly ou do Goal Engine foi alterada** — esta secção documenta apenas a ligação de módulos já existentes que, até agora, nunca eram chamados fora de testes (achado nº1 do sumário executivo, §3.2, §12.3).

### 15.1 Onde estava implementado, e porque não participava na produção

Recapitulando §3 (nada mudou nos ficheiros abaixo — apenas deixaram de ser código órfão):

- `src/engine/dixon_coles.py`: `tau()` (correção de dependência para scores baixos), `dixon_coles_simulate_match()` (matriz de probabilidades por resultado exato, Poisson bivariada + τ), `calculate_fractional_kelly()` (Kelly fracionário com cap de 2%). Recebe `lambda_home`/`mu_away` como argumentos externos — não calcula golos esperados a partir de dados, por desenho.
- `src/engine/value.py::evaluate_match_value()`: soma a matriz do Dixon-Coles por mercado 1X2 (`p_home`/`p_draw`/`p_away`) e calcula EV/stake (Fractional Kelly) por mercado.

**Onde é que a informação deixava de ser utilizada, exatamente:** `src/collector/client.py::EventCollector.get_matches()` construía cada `Match` com `xg_home=None, xg_away=None` (linhas 45-46, versão anterior) e uma única `probability` vinda de `predict_probability()` (heurística de H2H, §2.1) — nunca chamava `dixon_coles_simulate_match()` nem `evaluate_match_value()`. `src/cli/predict.py` (o único consumidor de `EventCollector` a partir de `main.py`) reutilizava essa mesma `match.probability` heurística para os três mercados (HOME/DRAW/AWAY) em vez de pedir uma probabilidade por mercado ao Dixon-Coles. É exactamente neste ponto — a construção do `Match` em `EventCollector.get_matches()` — que a cadeia "BSD API → probabilidades pré-jogo" parava de usar o modelo já implementado e passava a usar apenas a heurística de H2H.

### 15.2 O que foi acrescentado (adaptador de inputs, não um novo modelo)

`dixon_coles_simulate_match()` sempre exigiu `lambda_home`/`mu_away` como argumentos — e nenhum caminho de produção os fornecia (§3.2). Sem esses inputs, o modelo não pode ser invocado. Foi acrescentado o adaptador mínimo necessário, num ficheiro novo e isolado:

- **`src/engine/pregame_lambda.py::estimate_pregame_lambdas(h2h)`** (novo): reparte a média de golos por jogo já observada no histórico de confrontos diretos (`head_to_head.avg_total_goals`, o mesmo campo já lido por `predict_probability()`) entre as duas equipas, com uma inclinação limitada (`±15%`) pela diferença `home_win_rate − away_win_rate` e uma vantagem de casa fixa (`+6%` da repartição — mesma ordem de grandeza do `+3` fixo já usado em `predict_probability()`). Sem histórico de H2H, usa uma média de liga neutra (2.5 golos/jogo). Nunca lança exceção; nunca devolve λ abaixo de 0.35. **Não estima ataque/defesa por MLE, não faz decaimento temporal, não é uma calibração estatística** — é apenas a tradução dos dados de H2H já disponíveis para o formato que `dixon_coles_simulate_match()` sempre exigiu. A recomendação de estimação MLE completa (§12.2, §13 recomendação 3) continua por implementar — está fora do âmbito desta tarefa ("colocar o modelo em produção", não otimizá-lo).
- **`src/engine/value.py::estimate_pregame_probabilities(lambda_home, mu_away, rho)`** (novo): devolve `{"home", "draw", "away"}` (fração) chamando `dixon_coles_simulate_match()` — a mesma função já usada por `evaluate_match_value()` — sem exigir odds (ao contrário de `evaluate_match_value`, que também calcula EV/stake e por isso precisa das 3 odds em simultâneo). Para evitar duplicar a soma da matriz por mercado (`np.tril`/`np.trace`/`np.triu`), essa lógica foi extraída para `_market_probabilities_from_matrix()`, reutilizada tanto por `evaluate_match_value()` como por `estimate_pregame_probabilities()` — `evaluate_match_value()` produz exatamente os mesmos números de antes (coberto por teste de regressão, §15.4).

### 15.3 Onde a pipeline passou a usar o Dixon-Coles

- **`src/collector/client.py::EventCollector.get_matches()`**: para cada evento, calcula `lambda_home, mu_away = estimate_pregame_lambdas(h2h)` e `dixon_coles_probabilities = estimate_pregame_probabilities(lambda_home, mu_away)`, e passa a construir `Match(..., xg_home=lambda_home, xg_away=mu_away, dixon_coles_probabilities=dixon_coles_probabilities)`. Os campos `xg_home`/`xg_away` do `Match` — que existiam desde sempre no modelo de dados mas ficavam sempre a `None` — passam a ser preenchidos com os golos esperados do Dixon-Coles: é o ponto pedido pela tarefa em que "a pipeline necessita de golos esperados".
- **`src/models/match.py::Match`**: novo parâmetro opcional `dixon_coles_probabilities` (dict `home`/`draw`/`away`, fração 0.0–1.0), guardado como atributo e incluído em `to_dict()`. Assinatura anterior mantida 100% compatível (parâmetro novo, com default `None`, no fim da lista) — confirmado pelos scripts de demonstração existentes que chamam `Match(...)` posicionalmente (`src/tools/test_match.py`, `test_value_scanner.py`, `test_match_ranking.py`).
- **`src/cli/predict.py::run_predict()`**: em vez de `model_probability_fraction = match.probability / 100.0` (mesmo valor reutilizado para HOME/DRAW/AWAY), passa a usar `match.dixon_coles_probabilities[MARKET_TO_DIXON_COLES_KEY[market]]` — a probabilidade Dixon-Coles específica do mercado a ser avaliado nesse ciclo do loop. `calculate_edge()`/`calculate_ev()` (inalterados) passam a receber a probabilidade correta por mercado. `result["model_probability"]` (mostrado no relatório) passa a refletir essa mesma probabilidade, em vez de mostrar sempre a heurística de H2H independentemente do mercado. **Fallback:** se `match.dixon_coles_probabilities` for `None` ou não tiver a chave do mercado (ex.: falha ao calcular, dado inesperado), o código regressa à heurística de H2H (`match.probability / 100.0`) — a pipeline nunca falha por ausência de Dixon-Coles, seguindo o mesmo padrão defensivo já usado em `LivePipeline.calculate_dynamic_lambda()` (§4.3).
- `src/model/predictor.py::predict_probability()` (heurística de H2H) **não foi alterada nem removida** — continua a ser chamada (preenche `match.probability`, ainda mostrado no relatório e usado como fallback acima); deixou apenas de ser a fonte da probabilidade usada no cálculo de edge/EV quando o Dixon-Coles está disponível.

### 15.4 Testes adicionados

`tests/test_dixon_coles_pipeline.py` (23 testes):

- **Caracterização do núcleo Dixon-Coles** (`tau`, `dixon_coles_simulate_match`, `calculate_fractional_kelly`) — confirma que os valores conhecidos e a soma da matriz (=1.0) permanecem exatamente os mesmos, sem alteração de fórmula.
- **Regressão de `evaluate_match_value()`** — compara o resultado após o refactor (`_market_probabilities_from_matrix`) contra um cálculo manual da matriz, confirmando que `prob`/`ev`/`stake_pct` não mudaram.
- **`estimate_pregame_probabilities()`** — confirma que devolve exatamente as mesmas probabilidades que `evaluate_match_value()` para o mesmo par λ (mesma matriz subjacente), que somam 1.0, e que reage a λ mais forte com probabilidade mais alta.
- **`estimate_pregame_lambdas()`** — h2h ausente/vazio nunca lança exceção e usa o fallback de liga (2.5 golos); usa `avg_total_goals` quando presente; a vantagem de casa e o tilt de win-rate têm o sinal esperado e respeitam os limites (`MIN_LAMBDA`, `MAX_STRENGTH_TILT`).
- **`EventCollector.get_matches()`** (cliente HTTP e coletor de odds mockados, sem rede) — confirma que cada `Match` passa a ter `dixon_coles_probabilities` com as 3 chaves somando 1.0, e `xg_home`/`xg_away` preenchidos, com e sem histórico H2H.
- **`run_predict()`** (coletor mockado) — confirma que HOME/DRAW/AWAY recebem `model_probability` diferentes entre si (antes desta integração, os três eram sempre o mesmo número) e que o fallback para a heurística de H2H funciona quando `dixon_coles_probabilities` é `None`.

### 15.5 Caminhos mortos identificados e ação tomada

- **`src/collector/client_backup.py`** — cópia órfã e desatualizada de `src/collector/client.py` (sem `h2h_matches`, sem qualquer referência no resto do repositório). Sem nenhuma importação em código, testes ou workflows. **Removido** — seguro por não ter nenhum consumidor.
- **`src/engine/live_pipeline.py.bak`** — ficheiro `.bak` (não é um módulo Python importável), já identificado em §1.2 como não executado. É citado como prova histórica nesta própria auditoria (§1.2) e em `docs/AUDIT_BSD_401.md`. **Não removido** — inofensivo (não executável, não importado por nada) e mantido como registo do estado anterior do `LivePipeline`.
- **`src/engine/ranking.py::rank_bets()` / `src/engine/value_scanner.py::scan_value_opportunities()`** — caminho de pré-jogo alternativo que chama `analyze_bet(match.odds, match.probability)`, com a mesma limitação que existia em `cli/predict.py` antes desta integração (uma única probabilidade heurística reutilizada por todos os mercados). **Não usado por nenhum entry point de `main.py`** — apenas por `src/tools/test_value_scanner.py` (script de demonstração). Documentado aqui, não alterado: está fora do âmbito desta tarefa (que visa exclusivamente a pipeline `main.py predict`) e alterá-lo tocaria em `analyzer.py`/`filter.py`, que não foram auditados para esta integração.

### 15.6 Impacto esperado

- **Correção de um erro de mercado, não apenas uma "ligação" nova**: antes, `main.py predict` comparava a mesma probabilidade heurística de H2H contra as odds de HOME, DRAW e AWAY em simultâneo — matematicamente impossível de estar correta para os três mercados ao mesmo tempo (P(home)+P(draw)+P(away) da heurística nem sequer somava 1). Com o Dixon-Coles, cada mercado recebe a probabilidade que lhe corresponde na mesma distribuição conjunta (que soma exatamente 1.0 por construção, §3.1).
- **Golos esperados deixam de ser sempre `None`**: `xg_home`/`xg_away` no relatório de cada jogo (`Match.to_dict()`) passam a refletir os λ usados pelo Dixon-Coles, em vez de ficarem vazios.
- **Qualidade da probabilidade em si**: o Dixon-Coles em produção continua sujeito à mesma limitação já registada em §3.4/§12.2 — os λ vêm de um adaptador simples baseado em H2H (`avg_total_goals` + tilt de win-rate), não de uma estimação de ataque/defesa por MLE com decaimento temporal. A "meia fórmula" que faltava ao Dixon-Coles (§3.2) continua a faltar; o que mudou é que a "meia fórmula" que já existia (a distribuição de probabilidade dado λ) finalmente processa números reais em produção, em vez de nunca ser chamada. Não se espera, desta integração isolada, uma melhoria de calibração validada estatisticamente — isso exigiria o trabalho descrito em §12.2/§13 (fora do âmbito pedido: "não implementar melhorias ao próprio modelo").
- **Sem impacto no `main.py live`, no Monte Carlo, no Kelly ao vivo, no Goal Engine, nem no `run_backtest.py`** — nenhum destes caminhos foi tocado; confirmado por execução de `python main.py predict` (falha da mesma forma que antes, por falta de credenciais de API neste ambiente — comportamento idêntico ao baseline, sem novas exceções) e `python run_backtest.py --demo` (output numérico idêntico ao baseline) após as alterações.

---

## 16. Estimador de lambda estatisticamente mais forte (§3.2/§12.2 endereçados parcialmente)

**Data:** 2026-08-03. Âmbito: substituir o adaptador heurístico de
`lambda_home`/`mu_away` (§15.2) por um estimador que usa mais da
granularidade já devolvida pela API em `head_to_head`, com encolhimento
estatístico (shrinkage) para amostras pequenas. **Nenhuma fórmula do
Dixon-Coles (`tau`, `dixon_coles_simulate_match`, `calculate_fractional_kelly`),
do Monte Carlo, do Kelly ou do Goal Engine foi alterada** — mesma disciplina
de âmbito da secção 15. Documentação completa (metodologia, fórmulas,
assunções, limitações, validação): `docs/05_lambda_estimator.md`.

### 16.1 Porque este trabalho era necessário

A secção 3.2 já tinha identificado que `dixon_coles_simulate_match()` nunca
calculou λ a partir de dados — recebe-o sempre de fora. A secção 15
resolveu a metade "ligar o modelo à produção", usando o adaptador mais
simples possível (`pregame_lambda.py`, repartição de `avg_total_goals` por
uma partilha fixa + tilt de win-rate). A secção 12.2 já registava esta
simplificação como a limitação mais importante face a boas práticas
quantitativas: "ausência de estimação de parâmetros a partir de dados
históricos... o próprio repositório já tem uma função `decay.py::apply_exponential_decay`
pronta para isto, mas não está ligada ao Dixon-Coles". Esta secção liga
essa função, e usa dois campos de `head_to_head` que a API já devolve
(confirmado em `schema.yaml`) mas que nenhum código no repositório lia:
`home_goals`/`away_goals` (golos agregados por equipa no H2H) e
`recent_matches` (confrontos diretos individuais).

### 16.2 O que foi acrescentado — novo módulo, não reescrita do existente

- **`src/engine/lambda_estimator.py`** (novo): `estimate_lambda(h2h)` — mesmo
  contrato de `pregame_lambda.py::estimate_pregame_lambdas(h2h)` (nunca
  lança exceção, nunca devolve ≤0), mas escolhe o melhor nível de
  informação disponível (jogos recentes ponderados por recência > golos
  agregados por equipa > `avg_total_goals` inferido > prior de liga) e
  aplica encolhimento estatístico (`empirical-Bayes shrinkage`) proporcional
  à dimensão da amostra disponível. Delega explicitamente ao Nível C/D o
  adaptador já existente (`estimate_pregame_lambdas`), em vez de duplicar a
  lógica de tilt de win-rate/vantagem de casa uma segunda vez.
- **`src/engine/pregame_lambda.py`** — **ficheiro não alterado**. Continua a
  ser usado (a) como bloco de construção do Nível C/D acima, e (b) como
  fallback defensivo em `src/collector/client.py` caso o novo estimador
  levante alguma exceção inesperada. Todos os testes de §15.4 continuam a
  passar sem alteração.
- **`src/collector/client.py::EventCollector.get_matches()`**: passa a
  chamar `estimate_lambda(h2h)` (com `try/except` para
  `estimate_pregame_lambdas(h2h)` como rede de segurança) em vez de chamar
  diretamente o heurístico antigo. Nenhum outro consumidor a jusante
  (`Match`, `src/cli/predict.py`, `src/engine/value.py`) foi alterado — o
  contrato (dois floats > 0) manteve-se idêntico.

### 16.3 Validação

`scripts/benchmark_lambda_estimator.py` compara os dois estimadores em dois
planos: (1) cenários determinísticos lado-a-lado, e (2) uma simulação de
recuperação sintética — dados inteiramente sintéticos com verdade
fundamental conhecida por construção, **rotulada explicitamente como
validação da mecânica estatística, não como alegação de desempenho
preditivo real**. Resultado agregado dessa simulação (400 trials por
combinação cenário/tamanho de amostra, seed=42): MSE(λ) 0.655→0.331, Brier
0.2219→0.2203, Log Loss 0.6351→0.6312 (antigo→novo). A melhoria concentra-se
em amostras pequenas (1-5 jogos de H2H — o caso mais comum na prática); foi
também encontrada e documentada uma limitação genuína do novo estimador em
amostras grandes (o Nível A não continua a convergir indefinidamente, por
desenho — ver `docs/05_lambda_estimator.md` §3.3/§7.1).

**Backtest real (Brier/Log Loss/Calibração/ROI sobre jogos e odds reais):
não executado.** Confirma-se aqui a mesma conclusão já registada em §12.2 e
§15.6 — o repositório não tem, no momento desta alteração, um dataset que
ligue snapshots de `head_to_head` (tal como estariam disponíveis antes de
cada jogo, sem fuga de informação do resultado) a resultados finais e odds
reais. `examples/backtest/sample_real_games.csv` tem 9 linhas sem nenhum
par de equipas repetido e sem `head_to_head`; `data/live_history.db` só tem
snapshots ao vivo (§1.2, §8) e não está ligado a resultados finais de forma
utilizável aqui. Não há credenciais de API configuradas neste ambiente
(`src/config/settings.py::require_api_key()` confirma todas as variáveis de
ambiente suportadas como ausentes), pelo que também não foi possível puxar
um dataset novo da API ao vivo. O requisito exato de dados para fazer esta
validação no futuro está documentado em `docs/05_lambda_estimator.md`,
secção 7.2.

### 16.4 O que continua por fazer

Tal como a secção 15 não implementou a estimação de ataque/defesa por MLE
(§3.2), esta secção também não o faz — usa mais da granularidade do H2H já
disponível, não constrói uma nova fonte de dados de liga inteira. Continua
válida a recomendação nº1 do §13 nesse sentido específico, e fica registado
como novo requisito concreto (não estava explícito antes desta secção): uma
fonte de dados de golos por equipa ao longo de uma temporada completa (não
apenas confrontos diretos entre duas equipas específicas), hoje ausente de
qualquer caminho de recolha de dados do repositório.

---

## 17. Reauditoria de Edge (confirmação) e remoção de uma duplicação residual

**Data:** 2026-08-03. Âmbito: nova auditoria completa a todos os pontos do
repositório que calculam Edge, para confirmar se a unificação da secção 14
se mantém válida e se o bug de `src/engine/market.py` continua corrigido.
**Nenhuma fórmula do Dixon-Coles, Monte Carlo, λ, Kelly, Goal Engine,
Machine Learning ou Decision Engine foi alterada** — apenas chamadas ao
cálculo de Edge, conforme pedido.

### 17.1 Resultado da auditoria

Foram inspecionados todos os locais que referenciam "edge" no repositório
(`src/engine/edge.py`, `decision.py`, `live_decision.py`, `bet_engine.py`,
`analyzer.py`, `full_engine.py`, `market.py`, `cli/predict.py`,
`ranking.py`, `value_scanner.py`, `backtest/historical/evaluator.py`,
`scripts/live_scanner.py`, entre outros). Confirma-se que **existe
efetivamente apenas uma implementação oficial**, `calculate_edge()` em
`src/engine/edge.py` (secção 14), e que todos os módulos de produção já
identificados na secção 14.5 continuam a reutilizá-la sem recalcular a
fórmula localmente. O bug documentado em `src/engine/market.py` (§6.3/§14.4)
**continua corrigido**: `analyze_market()` chama
`calculate_edge(model_probability, odd)`, passando a odd e não a
probabilidade implícita de mercado.

### 17.2 Única divergência encontrada: `scripts/live_scanner.py`

Este script standalone (não invocado por nenhum entry point de `main.py`;
usado manualmente para scan ao vivo, ver `docs/AUDIT_BSD_401.md`)
recalculava a mesma fórmula localmente em vez de reutilizar
`calculate_edge`:

```python
implied = (1 / odd) * 100
edge = round(probability - implied, 2)
```

Matematicamente equivalente a `calculate_edge(probability/100, odd) * 100`
— não é um bug de valores incorretos, mas é exatamente o tipo de duplicação
de fórmula que a secção 14 já tinha eliminado dos módulos de produção.
**Corrigido**: o script passa a importar e chamar `calculate_edge`/
`implied_probability` de `src/engine/edge.py`, com uma guarda prévia
(`odd <= 1.0` ou `probability <= 0`) antes da chamada, no mesmo padrão
defensivo já usado por `DecisionEngine.evaluate_bet`, `evaluate_live_market`
e `bet_engine.evaluate_bet` (evita a exceção que `calculate_edge` agora
lança para inputs inválidos — ver §14.3 — em vez de deixar o erro cair no
`except Exception` genérico do script). Nenhum outro ficheiro precisou de
alteração.

### 17.3 Testes e impacto

`python -m pytest tests/` (206 testes) continua totalmente verde,
`python main.py predict` falha da mesma forma que na secção 15.6 (falta de
credenciais de API neste ambiente — comportamento de baseline, sem novas
exceções) e `python run_backtest.py --demo` produz o mesmo resumo numérico
de sempre. Impacto esperado desta secção: nenhuma mudança de comportamento
observável em produção (nenhum entry point de `main.py` importa
`scripts/live_scanner.py`); o único efeito é a remoção de uma fórmula de
Edge duplicada, mantendo `src/engine/edge.py::calculate_edge` como única
fonte oficial em todo o repositório.

---

## 18. Consolidação do label `goal_in_next_15m`

**Data:** 2026-08-03. Âmbito: auditoria e unificação de todas as
implementações da lógica de rotulagem `goal_in_next_15m` de
`data/live_history.db`, achado nº8 do sumário executivo (§8) e detalhado
em §10.2. **Nenhuma fórmula do Dixon-Coles, Monte Carlo, λ, Kelly, Goal
Engine, Machine Learning ou Decision Engine foi alterada** — apenas o
cálculo desta label, conforme pedido.

### 18.1 Confirmação da auditoria: três implementações, não quatro nem duas

A auditoria original (§10.2) continua exata. Foram confirmadas, por
inspeção direta do código e por pesquisa global de `goal_in_next_15m` em
todo o repositório, exatamente três implementações divergentes, todas já
descritas em §10.2:

| # | Ficheiro / função | Definição matemática | Corre em produção? |
|---|---|---|---|
| (a) | `src/backtest/logger.py::update_outcomes` (chamada por `src/engine/live_monitor.py`) | `UPDATE ... SET goal_in_next_15m = ? WHERE match_id=? AND current_minute BETWEEN (minuto_atual−18) AND (minuto_atual−12) AND goal_in_next_15m IS NULL` — janela de tolerância `[-18,-12]` à volta de "15 min atrás", incremental, só quando `NULL` | Sim, dentro de `live_monitor.py`, mas sempre sobrescrita por (b) no mesmo workflow |
| (b) | `src/training/create_labels.py` | `UPDATE ... SET goal_in_next_15m = CASE WHEN EXISTS (snapshot b do mesmo match_id com golos(b) > golos(atual) AND minuto(b) > minuto_atual AND minuto(b) <= minuto_atual+15) THEN 1 ELSE 0 END WHERE current_minute IS NOT NULL` — janela `(minuto, minuto+15]`, tabela inteira, sobrescreve sempre | Sim — passo "4.5" do workflow `live_logger.yml`, corre **por último**, por isso é sempre o valor final persistido |
| (c) | `src/backtest/labeler.py` | `UPDATE ... SET goal_in_next_15m = 1 WHERE id IN (snapshot a com snapshot posterior b do mesmo match_id, b.timestamp > a.timestamp AND b.timestamp <= a.timestamp+15min, com placar diferente)` — janela de 15 min sobre `timestamp` (relógio), só escreve `1` explicitamente | Não — script órfão, sem nenhum import nem chamada em workflows, testes ou outros módulos |

Diferenças entre as três: (a) usa uma janela de tolerância deslocada e
incremental sobre `current_minute`; (b) usa uma janela exata e fechada à
direita sobre `current_minute`, recalculada para toda a tabela; (c) usa
`timestamp` de relógio em vez de `current_minute`, e nunca atribui `0`
explicitamente. As três produzem resultados diferentes para os mesmos
dados sempre que a cadência real de `live_monitor.py` não é exatamente
"uma vez a cada 15 minutos" — e (a) e (c) nunca são as que ficam
persistidas, porque (b) corre depois de (a) no workflow e (c) nunca corre.

### 18.2 Implementação oficial escolhida

Escolhida a definição de **(b)**, agora centralizada em
`src/backtest/goal_label.py::recompute_goal_in_next_15m`:

```
goal_in_next_15m(s) = 1  se existe um snapshot posterior s' do mesmo
                          match_id, em minuto m', tal que
                              m < m' <= m + 15
                      e   golos(s') > golos(s)
                    = 0  caso contrário
```

(`golos(s) = home_score(s) + away_score(s)`; só se recalculam linhas com
`current_minute IS NOT NULL`.)

Razões para escolher esta definição, e não (a) ou (c):

1. **É a que já determinava o valor efetivamente persistido em produção.** Como (b) corre por último no workflow `live_logger.yml` e sobrescreve incondicionalmente, o conteúdo atual de `data/live_history.db` já reflete esta definição, não a de (a). Escolher qualquer outra implicaria reescrever o histórico existente (1683 snapshots, 75 jogos) com uma definição diferente da que gerou os dados já usados por `build_dataset.py`/`train_model.py`/`audit_dataset.py`.
2. **Não depende do relógio de execução do workflow.** (a) e (c) assumem, implícita ou explicitamente, uma cadência de execução regular (`[-18,-12]` em (a); `timestamp+15min` em (c)) — se `live_monitor.py` corre com atraso, corre duas vezes seguidas, ou falha uma execução, ambas produzem janelas erradas. (b) deriva inteiramente de `current_minute`, que é um dado do próprio jogo, não da cadência de execução do scraper.
3. **É determinística e idempotente**, ao contrário de (a) (`AND goal_in_next_15m IS NULL` faz depender o resultado da ordem/número de vezes que a função já correu) — confirmado nesta consolidação: recalcular (b) sobre uma cópia da `data/live_history.db` atual produz exatamente a mesma distribição (1135 negativos / 548 positivos) que já lá estava.

### 18.3 Consolidação aplicada

- **`src/backtest/goal_label.py`** (novo): única implementação oficial —
  `recompute_goal_in_next_15m(conn)` (usa uma ligação já aberta,
  não decide commit) e `recompute_goal_in_next_15m_for_db(db_path)`
  (conveniência para scripts, abre/faz commit/fecha).
- **`src/training/create_labels.py`**: deixou de ter o SQL embutido;
  passa a chamar `recompute_goal_in_next_15m(conn)`. Comportamento de CLI
  (mensagens impressas, distribuição final) inalterado.
- **`src/backtest/labeler.py`**: a lógica divergente baseada em
  `timestamp` foi substituída por uma chamada à mesma
  `recompute_goal_in_next_15m(conn)` — deixa de existir uma terceira
  fórmula, mesmo mantendo o ficheiro como ponto de entrada manual (não é
  chamado por nada em produção, tal como antes).
- **`src/backtest/logger.py`**: função `update_outcomes` **removida**.
  Era sempre sobrescrita por (b) no mesmo workflow (§10.2), pelo que o
  seu cálculo — a única divergência que corria de facto em produção antes
  de ser sobreposta — deixou de ter qualquer efeito observável no valor
  final; mantê-la seria manter código morto que recalcula uma fórmula
  incorreta sem propósito.
- **`src/engine/live_monitor.py`**: deixou de importar e chamar
  `update_outcomes` (ver ponto anterior). `init_db`/`log_snapshot`
  (schema e inserção de snapshots) não foram alterados.
- Nenhum outro consumidor da label precisou de alteração:
  `src/training/build_dataset.py`, `train_model.py`, `audit_dataset.py`
  e `scripts/app.py` apenas **leem** a coluna `goal_in_next_15m` já
  persistida — não a calculam — pelo que continuam a funcionar sem
  qualquer mudança de código.

### 18.4 Validação de compatibilidade

- **`data/live_history.db`**: recalcular `goal_in_next_15m` com a nova
  implementação sobre uma cópia da base de dados real (1683 snapshots, 75
  jogos) produz **exatamente a mesma distribuição** que já lá estava
  (1135 negativos / 548 positivos) — confirma que a consolidação não
  altera nenhum valor já persistido, apenas remove a duplicação de código
  que o gerava.
- **Treino** (`build_dataset.py` → `training_dataset.csv` →
  `train_model.py`): `build_dataset.py` foi executado sobre a base de
  dados real após a alteração e produz a mesma distribuição de
  `goal_in_next_15m` (1135/548) e o mesmo número de colunas; não depende
  de como a label foi calculada, só de já existir na tabela.
- **Backtesting** (`run_backtest.py --demo`): não usa
  `goal_in_next_15m` — o dataset histórico de `src/backtest/historical`
  deriva o resultado de mercado do placar final do jogo, não desta label
  ao vivo. Confirmado por pesquisa global: nenhum ficheiro em
  `src/backtest/historical/` referencia `goal_in_next_15m`. Execução
  `python run_backtest.py --demo` após a alteração produz o mesmo resumo
  numérico de sempre (8 jogos carregados, 3 apostas, ROI 53.33%).
- **Live monitor** (`src/engine/live_monitor.py`): continua a chamar
  `init_db`/`log_snapshot` sem alteração; deixou apenas de chamar
  `update_outcomes` (§18.3) — o recalculo da label passa a ficar
  inteiramente a cargo do passo "4.5" do workflow
  (`create_labels.py`, já existente, agora reutilizando a implementação
  central).

### 18.5 Testes adicionados

`tests/backtest/test_goal_label.py` (7 testes), sobre uma base de dados
sqlite em memória com o schema mínimo de `match_snapshots`:

- golo exatamente aos 15 minutos do snapshot (fronteira `m+15`, inclusive) → positivo;
- golo antes dos 15 minutos → positivo;
- golo depois dos 15 minutos (`m+16`) → negativo;
- vários golos dentro da janela → continua positivo (não conta golos, só existência);
- sem golos no jogo → negativo em todos os snapshots;
- golo já refletido no próprio snapshot (mesmo minuto) não conta como "dentro da janela" (a definição exige `m' > m`);
- linhas com `current_minute IS NULL` não são tocadas pelo recalculo.

### 18.6 Testes e impacto

`python -m pytest tests/` (213 testes, incluindo os 7 novos) totalmente
verde. `python main.py predict` falha da mesma forma que nas secções
15.6/17.3 (falta de credenciais de API neste ambiente — comportamento de
baseline, sem novas exceções; esta pipeline nunca referenciou
`goal_in_next_15m`). `python run_backtest.py --demo` produz o mesmo
resumo numérico de sempre (§18.4). Impacto esperado: nenhuma mudança de
valor observável em `data/live_history.db` nem nos pipelines de
treino/backtesting/live que a consomem; o único efeito é a remoção de
duas fórmulas divergentes (`update_outcomes` e a versão baseada em
`timestamp` de `labeler.py`), passando a existir uma única fonte de
verdade, `src/backtest/goal_label.py::recompute_goal_in_next_15m`,
reutilizada por todos os pontos de entrada que precisam de calcular esta
label.

---

## 19. Backtesting Framework de ponta a ponta sobre dados reais (Melhoria #4)

**Data:** 2026-08-04. Âmbito: colocar o Backtesting Framework
(`src/backtest/historical`) e o Framework de Avaliação (`src.evaluation`)
a correr de ponta a ponta sobre dados reais produzidos pelo Historical
Dataset Builder (`src.historical_dataset`) e pela BSD API — a lacuna já
registada em §16.3 ("Backtest real... não executado. O repositório não
tem, no momento desta alteração, um dataset que ligue... a resultados
finais e odds reais"). **Nenhuma fórmula do Dixon-Coles, Monte Carlo,
Kelly, Edge, EV, Goal Engine, Machine Learning ou Decision Engine foi
alterada** — mesma disciplina de âmbito das secções 15-18.

### 19.1 A peça em falta: `model_prob`

O Backtesting Framework (`load_historical_dataset`) e o Historical Dataset
Builder já existiam e já tinham uma ponte entre os dois
(`backtest_bridge.to_backtest_frame`) — mas essa ponte, deliberadamente,
nunca calculava a probabilidade do modelo (`model_prob`): exigia que quem
a chamasse a fornecesse. Como o Historical Dataset Builder não devolve
`head_to_head` por jogo (só a BSD API de eventos futuros o devolve), não
havia, antes desta secção, nenhum caminho automático que ligasse os dois
módulos sobre dados reais sem inventar a probabilidade à mão.

**Acrescentado** (`src/historical_dataset/backtest_bridge.py`), como
adaptador — não como novo modelo:

- **`derive_h2h(df, home_team, away_team, before)`**: reconstrói o dict
  `head_to_head` (mesmo formato que `estimate_lambda`/
  `estimate_pregame_lambdas` já esperavam de `EventCollector.get_matches()`
  para jogos futuros) a partir de confrontos diretos **anteriores** já
  presentes no próprio dataset devolvido pelo builder — sem pedir nada
  extra à BSD API. Sem fuga de informação: só entram jogos com data
  estritamente anterior ao jogo a avaliar. Reorienta os golos de cada
  confronto passado para a identidade do jogo a jogar (ver
  `docs/05_lambda_estimator.md` §5, ponto 1) quando as equipas jogaram
  com mandos de campo trocados.
- **`model_probabilities_from_dixon_coles(records)`**: para cada jogo,
  chama `derive_h2h` e depois o Dixon-Coles já em produção para jogos
  futuros (`src.engine.lambda_estimator.estimate_lambda` +
  `src.engine.value.estimate_pregame_probabilities`, secções 15/16 desta
  auditoria) — devolve `{event_id: {"home", "draw", "away"}}`.
- **`run_historical_backtest.py`** (novo, raiz do repositório): CLI que
  lê o CSV exportado por `build_historical_dataset.py`, calcula
  `model_prob` como acima para os mercados 1X2, converte via
  `to_backtest_frame`, corre `src.backtest.historical.BacktestEngine` e
  `src.evaluation.report.evaluate` (o Framework de Avaliação), e imprime/
  exporta o relatório completo (CSV, Markdown, Excel, HTML, gráficos).

Only os mercados HOME/DRAW/AWAY foram ligados — os únicos para os quais
`estimate_pregame_probabilities` já devolve probabilidade sem exigir
nenhuma agregação nova sobre a matriz de resultados do Dixon-Coles (Over/
Under e BTTS exigiriam somar essa matriz de outra forma, fora do âmbito
desta tarefa).

### 19.2 Incompatibilidade de esquema encontrada e corrigida (camada de conversão, não fórmula)

A primeira execução real revelou uma incompatibilidade genuína, nunca
antes exercitada: a BSD API devolve `event_date` em ISO 8601 UTC (ex.
`"2024-08-10T15:00:00Z"`), que `pandas.to_datetime` interpreta como
datetime **com fuso horário**. `openpyxl`/`pandas.ExcelWriter` (usados por
`BacktestReport.to_excel`, já existente) rejeitam datetimes com fuso
horário (`ValueError: Excel does not support datetimes with timezones`).
O `examples/backtest/sample_real_games.csv` ilustrativo usado por
`run_backtest.py --demo` só tem datas simples (`"2011-10-23"`, sem fuso),
por isso este caminho nunca tinha sido exercitado antes desta tarefa.

**Correção, só na camada de conversão** (`backtest_bridge._naive_dates`,
chamada por `to_backtest_frame`): normaliza a coluna `date` para UTC "sem
fuso" antes de entrar no Backtesting Framework — preserva o instante
exato (tudo já está em UTC) e só remove a anotação de fuso. Nem
`load_historical_dataset`, nem `BacktestEngine`, nem nenhum exportador
foram tocados. Teste de regressão em
`tests/historical_dataset/test_backtest_bridge.py::test_timezone_aware_dates_from_bsd_api_are_normalized_to_naive`.

### 19.3 Execução real — Historical Dataset Builder + Backtesting Framework + Framework de Avaliação

Executado em CI (GitHub Actions, com acesso real à BSD API via
`secrets.BZZOIRO_API_KEY` — este ambiente de auditoria não tem
credenciais nem acesso de rede à BSD API, a mesma limitação já registada
em §15.6/§16.3/§17.3/§18.6), workflow
`.github/workflows/run_historical_backtest.yml`
(`build_historical_dataset.py --competition-id 8` seguido de
`run_historical_backtest.py`), competição `id=8` "UEFA Europa League" —
a mesma já usada como referência em §16.3, por ser a única com cobertura
de odds reais já confirmada (época em curso). Execução completa em CI:
~3m30s.

| Métrica pedida | Valor |
|---|---|
| Jogos analisados (obtidos do Historical Dataset Builder) | 301 |
| Jogos com odd real e probabilidade válidas (pelo menos um mercado) | 30 |
| Apostas simuladas (jogo × mercado, HOME/DRAW/AWAY) | 90 |
| Apostas colocadas (`engine_decision=BET`, `DecisionEngine` real) | 35 |
| ROI | 24.06% |
| Yield | 24.06% |
| Profit (lucro líquido, stake fixo=1 unidade) | 8.42 |
| Odd média das apostas colocadas | 6.75 |
| Brier Score (todas as apostas avaliadas) | 0.20713 |
| Log Loss (todas as apostas avaliadas) | 0.6025 |
| Calibration Error / ECE (todas as apostas avaliadas) | 0.068431 |
| Max Drawdown | -11.22 (-201.44%) |
| Suite de testes completa (`python -m unittest discover -s tests`) | 444 testes, 0 falhas |

Dos 301 jogos devolvidos pelo builder, apenas 30 têm odds reais — confirma
em execução real (não apenas por inspeção estática, como em §16.3) a
limitação já documentada em `docs/07_historical_dataset_builder.md`
("Limitações", ponto 2): a BSD API só publica odds para a época em
curso/mais recente de cada competição; épocas já terminadas devolvem os
11 campos de odds a `null`. Este é o motivo do número relativamente baixo
de apostas simuladas face aos 301 jogos coletados — não uma falha do
bridge nem do Backtesting Framework.

**Sobre o Max Drawdown >100%:** `max_drawdown_pct` (`src/backtest/historical/metrics.py::max_drawdown`,
não alterado aqui) divide o drawdown máximo absoluto pelo pico de banca
acumulada **no momento em que esse drawdown ocorre** — com stake fixo=1 e
poucas dezenas de apostas, esse pico pode ser pequeno (ex. +1 ou +2
unidades acumuladas cedo na série) antes de uma sequência de perdas mais
longa, produzindo uma percentagem >100% em relação a esse pico específico
(não em relação a uma banca inicial fixa). É um comportamento conhecido
de dividir por um pico pequeno, não um bug introduzido por esta tarefa —
registado aqui para leitura correta do número, sem alterar a fórmula.

**Sobre a amostra ser pequena (30 jogos, 90 apostas) e o `model_prob` usado:**
com H2H derivado apenas dos confrontos diretos dentro do próprio dataset
(§19.1), a maioria dos jogos da época em curso não tem nenhum confronto
direto anterior registado nesta mesma execução (equipas que só se
cruzaram nesta época), pelo que `estimate_lambda` cai
predominantemente no prior de liga (Nível C/D, §16.1) — a mesma
limitação de fundo já registada em §16.4 (ausência de uma fonte de golos
por equipa ao longo de uma temporada completa) continua válida aqui; esta
tarefa liga o pipeline, não melhora a qualidade preditiva do estimador. O
ROI/Yield positivos observados (24.06%, N=35 apostas colocadas) não devem
ser lidos como evidência de valor preditivo real — a amostra é pequena
demais e o método de estimação de λ, nestas condições, é
predominantemente o prior de liga; ver §19.4.

### 19.4 O que continua por fazer

- Backtest sobre uma amostra maior (múltiplas competições/épocas em
  curso, não só uma) — limitado nesta tarefa pelo tempo de execução em CI
  e pela disciplina de âmbito ("colocar a funcionar", não "otimizar" nem
  "expandir a recolha de dados").
- Ligar Over/Under e BTTS (exigiria somar a matriz do Dixon-Coles de
  outra forma sobre `_market_probabilities_from_matrix` — fora do âmbito
  desta tarefa, que se limitou aos mercados já suportados por
  `estimate_pregame_probabilities` sem código novo de agregação).
- A limitação de fundo já registada em §12.2/§16.4 (falta de uma fonte de
  golos por equipa ao longo de uma temporada completa, para uma
  estimação de ataque/defesa por MLE) continua a ser a limitação mais
  relevante para a qualidade do `model_prob` usado neste backtest.

### 19.5 Ficheiros alterados/criados

- `src/historical_dataset/backtest_bridge.py`: `derive_h2h`,
  `model_probabilities_from_dixon_coles`, `_naive_dates` (novos); `to_backtest_frame`
  passou a normalizar `date` via `_naive_dates` (correção de esquema, §19.2).
- `run_historical_backtest.py` (novo): CLI de ponta a ponta.
- `.github/workflows/run_historical_backtest.yml` (novo): workflow manual
  (`workflow_dispatch`, igual em espírito a "Build Historical Dataset
  (BSD API)") que corre a suite de testes completa, constrói o dataset
  real e executa o backtest em CI.
- `tests/historical_dataset/test_backtest_bridge.py`: testes novos
  offline (`TestDeriveH2H`, `TestModelProbabilitiesFromDixonColes`, teste
  de normalização de datas com fuso horário) — 10 testes novos (de 12
  para 22).
- `docs/04_backtesting_framework.md`: nova secção com o exemplo real e a
  tabela de resultados desta execução.

---

## 20. Propagação da confiança do modelo até ao Evaluation Framework (Melhoria #8)

**Data:** 2026-08-04. Âmbito: o `LambdaEstimate` já produzido por
`src.engine.lambda_estimator.estimate_lambda_detailed` (Melhoria #5, §
anterior deste documento) já continha `tier` e `effective_sample_size` —
mas nenhum consumidor de produção usava a versão "detalhada": tanto
`src.collector.client` (jogos futuros) como
`src.historical_dataset.backtest_bridge.model_probabilities_from_dixon_coles`
(backtest) só chamavam `estimate_lambda(h2h) -> (lambda_home, mu_away)`,
descartando a proveniência da estimativa. Essa informação morria em
`LambdaEstimate` e nunca chegava a `HistoricalBet` nem ao Framework de
Avaliação (`src.evaluation`), impedindo medir o desempenho por nível real
de confiança do modelo (só era possível segmentar por `probability` — a
confiança na SELEÇÃO apostada, não na estimativa de λ que a produziu).
**Nenhuma fórmula do Dixon-Coles, Monte Carlo, Kelly, Edge, EV, Goal
Engine, Machine Learning ou Decision Engine foi alterada** — mesma
disciplina de âmbito das secções 15-19; esta melhoria é estritamente
aditiva (metadados opcionais).

### 20.1 O que foi acrescentado

- **`src/engine/lambda_estimator.py`**: `classify_model_confidence(tier,
  effective_sample_size) -> "HIGH"|"MEDIUM"|"LOW"` (novo). Categoriza a
  proveniência da estimativa reutilizando o mesmo `SHRINKAGE_K` já
  definido para o encolhimento estatístico (nenhuma constante nova sem
  justificação). Nunca é chamada pelo motor de decisão nem influencia
  `lambda_home`/`mu_away`.
- **`src/historical_dataset/backtest_bridge.py`**:
  `lambda_confidence_from_dixon_coles(records)` (novo) — espelha
  `model_probabilities_from_dixon_coles` (mesmo `derive_h2h`, sem fuga de
  informação nova) mas chama `estimate_lambda_detailed` em vez de
  `estimate_lambda`, devolvendo `{event_id: {"lambda_tier",
  "effective_sample_size", "model_confidence"}}`. `to_backtest_frame`
  ganhou três parâmetros opcionais (`lambda_tier`,
  `effective_sample_size`, `model_confidence`, mesmo mecanismo de
  resolução que `model_prob`) — omitidos, o DataFrame devolvido fica
  exatamente como antes (retrocompatível).
- **`src/backtest/historical/models.py`**: `HistoricalBet` e
  `EvaluatedBet` ganharam os três campos opcionais (`model_confidence`,
  `lambda_tier`, `effective_sample_size`, todos `None` por omissão).
  `HistoricalBet.from_dict` aceita-os por alias (PT/EN) e trata `NaN`
  como ausência (não como dado) — um CSV/DataFrame sem estas colunas
  continua a validar exatamente como antes.
- **`src/backtest/historical/evaluator.py`**: `evaluate_bet` passa estes
  três campos de `HistoricalBet` para `EvaluatedBet` sem os tocar —
  nunca entram no cálculo de `probability`/`edge`/`ev`/`kelly`/`stake`.
- **`src/evaluation/segments.py`**: três segmentos novos —
  `segment_by_lambda_tier`, `segment_by_model_confidence`,
  `segment_by_effective_sample_size_range` (bins `0-2 | 2-4 | 4-8 | 8-15 |
  15+`, à volta de `SHRINKAGE_K=4`) — e `all_confidence_segments`, que os
  agrega. Ao contrário dos segmentos "clássicos" (só financeiro, sobre
  `placed_bets`), estes combinam ROI/Yield/Nº de apostas (sobre as
  apostas colocadas de cada grupo) com Brier Score/Log Loss (sobre TODAS
  as apostas avaliadas desse grupo — mesma convenção de
  `evaluation.metrics.full_summary`), porque precisam de `all_bets`.
  Grupos sem o metadado disponível são omitidos, nunca geram erro.
- **`src/evaluation/report.py`**: `EvaluationReport.from_backtest_report`
  passou a incluir `all_confidence_segments(report.all_bets)` em
  `extra_segments` — os três segmentos novos aparecem automaticamente em
  TODAS as exportações já existentes (CSV, Excel, HTML, Markdown), sem
  qualquer alteração a essas funções de exportação.
- **`run_historical_backtest.py`**: passou a calcular
  `lambda_confidence_from_dixon_coles(records)` a par de
  `model_probabilities_from_dixon_coles`, a alimentar
  `to_backtest_frame(...)` com os três metadados, e a imprimir a tabela
  `by_lambda_tier` (ROI/Yield/Brier/Log Loss/Nº de apostas por tier).

### 20.2 Retrocompatibilidade

- `HistoricalBet.from_dict` sobre um dict/linha sem estes campos devolve
  os três atributos a `None`, sem erro (`tests/test_confidence_propagation.py::TestHistoricalBetConfidenceFields`).
  `NaN` (coluna presente num DataFrame mas vazia nessa linha) é tratado
  da mesma forma que ausência.
- `EvaluatedBet.to_dict()` emite sempre as três chaves (a `None` quando
  não fornecidas) — a coluna existe no DataFrame de `all_bets`/`placed_bets`
  produzido por `evaluate_bets`, mas fica vazia; os segmentos novos
  detetam isso e omitem-se automaticamente (`all_confidence_segments`
  devolve `{}` quando não há nenhum valor não-nulo).
- `examples/backtest/sample_real_games.csv` (sem estas colunas) e o
  dataset sintético de `tests.backtest.fixtures.generate_sample_dataset`
  continuam a produzir exatamente os mesmos `global_metrics`/
  `statistical_metrics` de sempre — `python run_backtest.py --demo`
  produz o mesmo resumo (ROI 53.33%, N=3 apostas colocadas) que antes
  desta melhoria.

### 20.3 Testes adicionados

`tests/test_confidence_propagation.py` (39 testes): `classify_model_confidence`
(bandas HIGH/MEDIUM/LOW, incluindo amostra `None`/`NaN`/negativa);
retrocompatibilidade de `HistoricalBet`/`EvaluatedBet`; propagação ponta a
ponta (`lambda_confidence_from_dixon_coles` → `to_backtest_frame` →
`load_historical_dataset` → `BacktestEngine` → `EvaluationReport`);
segmentação correta por `lambda_tier`/`model_confidence`/faixa de
`effective_sample_size` (financeiro só sobre colocadas, estatístico sobre
todas as avaliadas do grupo, valores calculados à mão); e ausência de
regressões (dataset sintético de 80 jogos e o CSV de exemplo legado
continuam a produzir os mesmos `global_metrics`/`statistical_metrics` e
não geram os segmentos novos, por não terem o metadado).

### 20.4 Testes e impacto

`python -m pytest tests/` — 525 testes, 0 falhas (486 já existentes + 39
novos). `python run_backtest.py --demo` produz exatamente o mesmo
resumo de sempre (ROI 53.33%, N=3 apostas colocadas, §19.4) — confirma
que nada no Backtesting Framework original foi alterado. Impacto
esperado: nenhuma mudança de valor em nenhuma métrica/decisão já
existente (ROI, Yield, Brier, Log Loss, Edge, EV, Kelly, decisão do
motor); o único efeito observável é a disponibilidade de três colunas
opcionais (`model_confidence`, `lambda_tier`, `effective_sample_size`) e
de três segmentos novos no Framework de Avaliação, ambos vazios/ausentes
sempre que a fonte de dados não fornece o metadado.

---

## 21. Kelly fracionário escalado pela confiança do modelo (Melhoria #6)

**Data:** 2026-08-04. Âmbito: §7 desta auditoria identificou três
implementações independentes de Kelly fracionário (`kelly.py`,
`dixon_coles.py::calculate_fractional_kelly`,
`decision.py::DecisionEngine.evaluate_bet`), cada uma com a fórmula de
Kelly completo recalculada localmente e a mesma fração fixa (0.25, "1/4
Kelly") hard-coded de forma independente — e notou (§7.5) que "não há, em
nenhuma das implementações, ajuste de Kelly pela incerteza do modelo,
apesar de haver informação de confiança disponível". Essa informação
(`LambdaEstimate.tier`/`.effective_sample_size`, Melhoria #5, já
propagada até `HistoricalBet`/`EvaluatedBet` pela Melhoria #8) continuava
sem qualquer efeito no tamanho da aposta. **Nenhuma fórmula de
Dixon-Coles, Monte Carlo, Goal Engine, Machine Learning, Edge ou EV foi
alterada, e nenhum critério de seleção de aposta (`min_edge`,
`engine_decision`, `placed`) foi tocado** — esta melhoria afeta
exclusivamente a FRAÇÃO de Kelly usada para dimensionar o stake de uma
aposta já decidida.

### 21.1 O que foi acrescentado

- **`src/engine/kelly.py`**: duas funções novas, chamadas por todas as
  implementações de Kelly do projeto —
  `calculate_confidence_multiplier(lambda_tier, effective_sample_size)` e
  `calculate_adaptive_kelly_fraction(base_fraction, lambda_tier,
  effective_sample_size)`. Fórmula:

      confidence_multiplier = n_eff / (n_eff + k)
      k = SHRINKAGE_K * peso_do_tier   (SHRINKAGE_K=4.0, já existente em lambda_estimator.py)
      adaptive_fraction = base_fraction * confidence_multiplier

  `peso_do_tier` depende apenas do valor (fixo, não de `n_eff`) de
  `LambdaEstimate.tier`: `recent_matches` (Nível A) -> peso 1 (`k=4`);
  `h2h_goal_totals` (Nível B) -> peso 2 (`k=8`); `avg_total_goals_or_prior`
  (Nível C/D, ou qualquer tier desconhecido) -> peso 4 (`k=16`), a mesma
  hierarquia de qualidade de informação já documentada na cascata de
  `estimate_lambda_detailed` (§16). Propositadamente **não** reutiliza
  `classify_model_confidence` (Melhoria #8) para esta escala: essa função
  tem fronteiras rígidas sobre `effective_sample_size` (muda de rótulo
  exatamente em `n_eff==8`, `==4`, `==2`), e usá-la introduziria um salto
  descontínuo do multiplicador exatamente nessas fronteiras — violando o
  requisito de continuidade. Ao depender só de `lambda_tier` (fixo por
  estimativa) para escolher `k`, o multiplicador é uma função contínua e
  suave de `effective_sample_size` em toda a gama.

  A forma `n/(n+k)` é a mesma família já usada em
  `lambda_estimator.py::_shrink_to_prior` — não introduz uma fórmula nova
  ao projeto, reaplica-a com o mesmo `SHRINKAGE_K` como referência.
  Propriedades: contínua e limitada a `[0, 1)` em `n_eff`; `n_eff=0` ->
  `0` (multiplicador mínimo); `n_eff -> infinito` -> `multiplicador -> 1`
  (nunca ultrapassa `1`, logo `adaptive_fraction` nunca ultrapassa
  `base_fraction`); sem `lambda_tier` OU `effective_sample_size`
  (omissos) -> `multiplicador = 1.0` exatamente, ou seja
  `adaptive_fraction == base_fraction`.

- **`src/engine/kelly.py::fractional_kelly`**: ganhou dois parâmetros
  opcionais (`lambda_tier=None`, `effective_sample_size=None`), passados
  a `calculate_adaptive_kelly_fraction` em vez de usar `fraction`
  diretamente. Omissos, comportamento idêntico ao de antes.

- **`src/engine/dixon_coles.py::calculate_fractional_kelly`**: deixou de
  recalcular a fórmula de Kelly completo localmente — passa a chamar
  `src.engine.kelly.kelly_fraction` e
  `calculate_adaptive_kelly_fraction`, eliminando a duplicação
  identificada em §7.2. Ganhou os mesmos dois parâmetros opcionais; o
  cap explícito (`max_stake_pct`, 2% por omissão) continua aplicado
  exatamente da mesma forma, sobre o resultado já escalado.

- **`src/engine/decision.py::DecisionEngine.evaluate_bet`**: deixou de
  recalcular `full_kelly` inline — passa a chamar
  `src.engine.kelly.kelly_fraction`. Ganhou os mesmos dois parâmetros
  opcionais, usados para escalar `self.max_kelly_fraction` via
  `calculate_adaptive_kelly_fraction` antes de multiplicar pelo Kelly
  completo. O cap explícito de 5% (§7.3) e o critério `edge >= min_edge`
  continuam exatamente iguais — a confiança nunca decide BET/PASS, só o
  tamanho do stake quando já é BET.

- **`src/backtest/historical/staking.py::KellyStake.stake_for`** (e a
  interface `StakingStrategy.stake_for`): ganhou os mesmos dois
  parâmetros opcionais, repassados a `fractional_kelly`. `FlatStake`
  ignora-os (stake fixo, por definição independente de Kelly/confiança).

- **`src/backtest/historical/evaluator.py::evaluate_bet`**: passa a
  propagar `bet.lambda_tier`/`bet.effective_sample_size` (já presentes em
  `HistoricalBet` desde a Melhoria #8, até agora só usados para
  segmentação) para `staking.stake_for(...)` — o único efeito é no campo
  `stake` quando a estratégia usada é `KellyStake`; o campo `kelly`
  (Kelly completo, sem fração) e `probability`/`edge`/`ev`/
  `engine_decision`/`placed` não são tocados.

### 21.2 Fórmula — comparação Kelly antigo vs. novo

```
Antigo (qualquer das três implementações):
    stake_fraction = kelly_full * base_fraction              (base_fraction fixo, ex. 0.25)

Novo (Melhoria #6):
    confidence_multiplier = n_eff / (n_eff + k(tier))        (k(tier) in {4, 8, 16})
    adaptive_fraction      = base_fraction * confidence_multiplier
    stake_fraction         = kelly_full * adaptive_fraction

Sem metadados de confiança (tier/n_eff == None):
    confidence_multiplier = 1.0  =>  adaptive_fraction == base_fraction
    => stake_fraction idêntico ao "Antigo", byte a byte.
```

Exemplo numérico (`probability=0.55`, `odd=2.10`, `base_fraction=0.25`,
`kelly_full≈0.1409`): sem metadados, `stake_fraction≈3.52%` da banca
(igual ao valor antes desta melhoria, `src/tools/test_kelly.py`). Com
`tier="avg_total_goals_or_prior"` e `n_eff=1` (amostra efetiva muito
pequena, pior tier): `confidence_multiplier≈0.059`, `stake_fraction≈0.21%`
— um stake muito mais conservador para uma estimativa pouco confiável.
Com `tier="recent_matches"` e `n_eff=1 000 000` (amostra efetiva enorme,
melhor tier): `confidence_multiplier≈0.999996`, `stake_fraction≈3.52%` —
praticamente indistinguível do Kelly fixo, como esperado (amostra grande
o suficiente para a confiança deixar de ser o fator limitante).

### 21.3 Retrocompatibilidade

- Todos os parâmetros novos (`lambda_tier`, `effective_sample_size`) são
  opcionais, com omissão (`None`) por defeito, em toda a cadeia
  (`kelly.py`, `dixon_coles.py`, `decision.py`, `staking.py`). Nenhuma
  assinatura pública existente foi quebrada — todas as chamadas
  existentes no repositório (`bet_engine.py`, `value.py`,
  `full_engine.py`, `report/dashboard.py`, `backtest/historical/
  evaluator.py` sem os novos argumentos, etc.) continuam a funcionar sem
  alteração.
- Confirmado por teste (`tests/test_adaptive_kelly.py`) e por execução
  real: `python run_backtest.py --demo` continua a produzir exatamente o
  mesmo resumo de sempre (ROI 53.33%, N=3 apostas colocadas) — a
  estratégia de staking por omissão do demo é `FlatStake`, que ignora
  Kelly por completo, e mesmo a via `KellyStake` produz o valor de
  sempre quando `HistoricalBet` não traz `lambda_tier`/
  `effective_sample_size` (caso do dataset de exemplo e do dataset
  sintético usados nos testes existentes).
- `src/report/dashboard.py` (produção, `main.py live`) **não foi
  alterado** — continua a chamar `DecisionEngine.evaluate_bet(market,
  prob, odd)` sem os novos argumentos, logo continua a produzir
  exatamente os mesmos stakes de sempre. A escala por confiança só entra
  em vigor onde um chamador passa `lambda_tier`/`effective_sample_size`
  explicitamente (atualmente, apenas o Backtesting Framework, via
  `evaluate_bet`/`KellyStake`).

### 21.4 Testes adicionados

`tests/test_adaptive_kelly.py` (38 testes): multiplicador de confiança
sem metadados (`None`/NaN/valor inválido -> `1.0`, sem escala);
`effective_sample_size` muito pequeno vs. elevado; tier
`recent_matches`/`h2h_goal_totals`/`avg_total_goals_or_prior` (o
"HIGH"/"MEDIUM"/"LOW" desta melhoria) e a sua ordenação a igual amostra;
continuidade (varrimento fino de `effective_sample_size`, sem saltos, em
particular à volta das fronteiras onde `classify_model_confidence`
mudaria de rótulo); limites (`0 <= adaptive_fraction <= base_fraction`
sempre); regressão de `fractional_kelly`,
`dixon_coles.calculate_fractional_kelly`,
`DecisionEngine.evaluate_bet` e `KellyStake`/`evaluate_bet` do
Backtesting Framework (valor sem metadados idêntico ao documentado em
§7); consistência cruzada entre as três implementações (mesmo resultado,
com e sem metadados); e confirmação de que confiança nunca afeta
`probability`/`edge`/`ev`/`engine_decision`/`placed`, só `stake`.

### 21.5 Testes e impacto

`python -m pytest tests/` — 563 testes, 0 falhas (525 já existentes + 38
novos). `python run_backtest.py --demo` produz exatamente o mesmo resumo
de sempre (ROI 53.33%, N=3 apostas colocadas, §19.4/§20.4). Impacto
esperado: nenhuma mudança em nenhum caminho de produção existente
(`main.py live`/`dashboard.py` não alterado); no Backtesting Framework,
quando `KellyStake` é combinado com apostas que trazem
`lambda_tier`/`effective_sample_size` (Melhoria #8), o stake de apostas
de baixa confiança passa a ser automaticamente reduzido (nunca
aumentado) face ao Kelly fixo — o efeito esperado é uma exposição de
banca menor, e portanto **drawdown máximo esperado menor ou igual** ao
de Kelly fixo, nos períodos/segmentos em que o modelo está a operar com
menos confiança (amostra pequena e/ou tier fraco), sem alterar quais
apostas são colocadas nem o seu Edge/EV.

---

## 22. Remoção do overround antes do cálculo de Edge/EV (Melhoria #7)

**Data:** 2026-08-04. Âmbito: `src/engine/edge.py`, `src/engine/market.py`,
`src/cli/predict.py`, `tests/test_edge.py`, `tests/test_overround.py`.
Endereça o achado de §6.4: *"o sistema nunca remove o overround ... antes
de calcular edge — o que significa que o 'edge' calculado contra qualquer
mercado individual já está a comparar a probabilidade do modelo contra
uma probabilidade de mercado inflacionada pela margem"*.

### 22.1 Diagnóstico

`implied_probability(odd) = 1/odd` é a probabilidade implícita de **uma
odd isolada** — inclui sempre a margem da casa (overround), porque a
soma das probabilidades implícitas de todas as opções de um mercado
(ex. as 3 odds de um 1X2) é sistematicamente > 1.0 (tipicamente 1.02 a
1.10 em mercados líquidos). `calculate_edge()` usava sempre esta
probabilidade "suja" (com margem) como referência de mercado — nunca a
probabilidade "fair" (sem margem) que se obtém normalizando as
probabilidades implícitas de todas as opções do mercado para somarem
1.0. Isto tornava sistematicamente mais difícil "ter edge" do que
deveria (barra artificialmente mais alta), sem que essa fosse uma
decisão de design documentada.

`calculate_ev(prob_model, odd_house) = prob_model * odd_house - 1` **não
usa `implied_probability` em nenhum ponto da sua fórmula** — depende
apenas da probabilidade do próprio modelo e da odd real paga pela casa.
Por isso, e como já registado em §6.1/§11 com confiança "Alto", a
correção matemática da fórmula de EV em si **não é afetada** pela
remoção do overround: o EV de apostar a `odd_house` com uma
probabilidade `prob_model` é o mesmo, quer o overround seja ou não
removido de outras odds do mesmo mercado.

### 22.2 Método escolhido para remover o overround

Normalização proporcional das probabilidades implícitas (o método básico
e mais comum, também usado por `src/backtest/market.py` na direção
inversa — ver §6.5):

```
overround            = Σ implied_probability(odd_i), para todas as opções i do mercado
fair_probability(i)  = implied_probability(odd_i) / overround
```

Implementado como função única e reutilizável, `remove_overround()`
(`src/engine/edge.py`), que aceita as odds de um mercado como `dict`
(`outcome -> odd`) ou lista, preservando a forma recebida e devolvendo
`None` quando há menos de 2 odds válidas (`odd > 1.0`) — sinal explícito
para o chamador manter o comportamento anterior.

### 22.3 Alterações a `calculate_edge()`/`calculate_ev()`

Ambas passaram a aceitar um terceiro parâmetro **opcional**,
`market_odds` (todas as odds do mesmo mercado, incluindo `odd_house`):

- `calculate_edge(prob_model, odd_house, market_odds=None)`: quando
  `market_odds` tem pelo menos 2 odds válidas (incluindo `odd_house`), o
  overround é removido via `remove_overround()` e a probabilidade fair
  resultante substitui `implied_probability(odd_house)` no cálculo do
  edge. Caso contrário (parâmetro omitido, odds insuficientes, ou
  `odd_house` ausente do conjunto), o resultado é **exatamente** o
  mesmo de antes desta melhoria.
- `calculate_ev(prob_model, odd_house, market_odds=None)`: aceita o
  mesmo parâmetro, por simetria de interface com `calculate_edge()` (para
  que um chamador com o mercado completo o possa passar a ambas as
  funções sem tratamento especial), mas **não o usa** no cálculo — o
  valor devolvido é idêntico com ou sem `market_odds` (ver §22.1 e
  `tests/test_overround.py::TestCalculateEvUnaffectedByOverroundRemoval`).

Nenhuma assinatura existente foi quebrada: quem continua a chamar
`calculate_edge(prob_model, odd_house)`/`calculate_ev(prob_model,
odd_house)` com dois argumentos posicionais obtém o comportamento
anterior, sem alterações.

### 22.4 Módulos que passam a reutilizar `market_odds`

- **`src/engine/market.py::analyze_market()`**: já recebia, por
  definição, o conjunto completo das odds de um mercado (`odds: dict`)
  — passa agora esse mesmo dict como `market_odds` a `calculate_edge()`/
  `calculate_ev()`, para os 3 tipos de mercado cobertos pelos testes
  (1X2, Over/Under, BTTS — a função é agnóstica ao tipo de mercado, só
  processa o dict de odds recebido). O campo `"market_probability"`
  devolvido mantém-se **inalterado** (probabilidade implícita simples,
  com margem) para não quebrar consumidores existentes desse campo
  (`src/report/printer.py`, `src/engine/report.py`); só o campo `"edge"`
  passa a refletir a probabilidade fair quando há mercado suficiente.
- **`src/cli/predict.py::run_predict()`** (caminho ativo de `main.py
  predict`): já tinha, por jogo, as 3 odds HOME/DRAW/AWAY disponíveis em
  `match.odds` — passa a construir `market_odds_1x2` a partir dessas
  mesmas odds (sem nenhuma chamada HTTP nova) e a passá-lo a
  `calculate_edge()`/`calculate_ev()` para cada um dos 3 mercados.

Nenhum outro módulo foi alterado: `src/engine/decision.py`
(`DecisionEngine`, usado em `main.py live`), `src/engine/live_decision.py`,
`src/engine/bet_engine.py`, `src/engine/analyzer.py`,
`src/backtest/historical/evaluator.py` e `src/engine/kelly.py` continuam
a chamar `calculate_edge`/`calculate_ev` com dois argumentos (sem
`market_odds`), porque não têm — nesses caminhos — o conjunto completo
de odds do mesmo mercado disponível de forma trivial (ex. `HistoricalBet`
só guarda a odd da aposta efetivamente registada, não as odds das outras
opções do mesmo mercado nesse jogo). Continuam, portanto, a produzir
exatamente os mesmos números de sempre — Dixon-Coles, Monte Carlo, Goal
Engine, Machine Learning, Kelly, Dashboard e os critérios de seleção de
apostas (thresholds de decisão) não foram tocados por esta melhoria.

### 22.5 Testes adicionados

`tests/test_overround.py` (34 testes) + 2 testes atualizados em
`tests/test_edge.py`:

- `remove_overround()`: soma das probabilidades fair = 1.0; probabilidade
  fair sempre menor que a implícita quando há margem; preserva a forma
  (dict/lista) e as chaves recebidas; `None` com 0 ou 1 odds válidas
  (ausência de odds suficientes);
- mercado 1X2 (3 odds, margem ~6%) com valores calculados à mão;
- mercado Over/Under (2 odds simétricas) — probabilidades fair exatas de
  0.5/0.5;
- mercado BTTS (Sim/Não) com valores calculados à mão;
- overround elevado (~20%) vs. overround baixo (~2%): a mesma
  probabilidade fair subjacente (0.5) produz o mesmo edge fair
  independentemente da margem, enquanto o edge antigo (sem remoção)
  varia com a margem — demonstra o efeito concreto da correção;
- ausência de odds suficientes: `market_odds` omitido, vazio, com uma só
  odd válida, ou sem a `odd_house` no conjunto — todos caem no fallback
  documentado (comportamento idêntico ao anterior);
- retrocompatibilidade: chamadas de 2 argumentos reproduzem exatamente os
  fixtures pré-existentes (`edge=0.0738`, `ev=0.155` para `p=0.55,
  odd=2.10`); validações de erro (`ValueError`/sentinela `-1.0`)
  inalteradas;
- regressão: `DecisionEngine`, `evaluate_live_market`, `bet_engine`,
  `analyzer.analyze_bet` e `Kelly` continuam a produzir os mesmos valores
  de sempre (não passam `market_odds`);
- `analyze_market()` e `run_predict()`: edge por mercado (1X2/O-U/BTTS)
  coincide com `calculate_edge(..., market_odds=...)`; EV inalterado;
  `market_probability` inalterado; mercado de 1 outcome cai no fallback.

### 22.6 Testes e impacto

`python -m pytest tests/` — 606 testes, 0 falhas (563 já existentes + 43
novos: 34 em `tests/test_overround.py`, mais 9 provenientes da expansão
das classes de teste já existentes em `tests/test_edge.py`). Impacto
esperado: para mercados com pelo menos 2 odds válidas disponíveis em
simultâneo (1X2 completo no `main.py predict`, ou qualquer mercado
passado a `analyze_market()`), o Edge calculado passa a ser
sistematicamente **maior ou igual** ao anterior (a probabilidade fair é
sempre ≤ probabilidade implícita com margem), reduzindo a barra
artificial identificada em §6.4 e tornando o Edge mais fiel à
probabilidade real de mercado — sem alterar o EV, o Kelly, a Decision
Engine, o Dashboard, ou qualquer critério de seleção de apostas em
produção (`main.py live` continua a não passar `market_odds`, logo
continua byte-a-byte igual a antes desta melhoria).
