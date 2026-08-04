# Relatorio de Auditoria de Dataset - Deteccao de Leakage

Dataset analisado: `data/training_dataset.csv`

## 1. Visao geral

- Linhas: 2013
- Colunas: 13
- Coluna alvo: `goal_in_next_15m`

### Colunas e tipos

| Coluna | Tipo |
|---|---|
| match_id | int64 |
| current_minute | int64 |
| home_score | int64 |
| away_score | int64 |
| dangerous_attacks_10m | int64 |
| shots_on_target_10m | int64 |
| corners_10m | int64 |
| live_odd_over | float64 |
| pressure | float64 |
| live_xg | float64 |
| red_cards | int64 |
| possession | float64 |
| goal_in_next_15m | int64 |

## 2. Estatisticas por feature

| Feature | Min | Max | Media | Std | Nº unicos |
|---|---|---|---|---|---|
| match_id | 46417.0 | 224849.0 | 208381.35370094387 | 36930.49332750115 | 96 |
| current_minute | 0.0 | 115.0 | 43.38748137108793 | 28.39500352026358 | 105 |
| home_score | 0.0 | 7.0 | 0.8688524590163934 | 1.0759842341582988 | 8 |
| away_score | 0.0 | 6.0 | 0.6572280178837556 | 0.9087292955410038 | 7 |
| dangerous_attacks_10m | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| shots_on_target_10m | 1.0 | 1.0 | 1.0 | 0.0 | 1 |
| corners_10m | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| live_odd_over | 1.85 | 1.85 | 1.8500000000000005 | 4.441995562823409e-16 | 1 |
| pressure | 10.55 | 40.0 | 24.42507699950323 | 7.468258774426722 | 243 |
| live_xg | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| red_cards | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| possession | 50.0 | 50.0 | 50.0 | 0.0 | 1 |

## 3. Correlacao absoluta com o alvo

| Feature | Correlacao absoluta |
|---|---|
| home_score | 0.06658419919759553 |
| pressure | 0.06477707549188404 |
| current_minute | 0.05105305616353267 |
| away_score | 0.03315778510956776 |
| match_id | 0.00010404939489013618 |
| live_odd_over | 1.1814259785983716e-16 |
| dangerous_attacks_10m | None |
| shots_on_target_10m | None |
| corners_10m | None |
| live_xg | None |
| red_cards | None |
| possession | None |

## 4. AUC por feature isolada (RandomForest)

Limiar de leakage: AUC > 0.9

| Feature | AUC | Alerta |
|---|---|---|
| match_id | 0.6754345711191468 |  |
| pressure | 0.6456432753905175 |  |
| current_minute | 0.6285541217716918 |  |
| home_score | 0.5416766093357074 |  |
| dangerous_attacks_10m | 0.5 |  |
| shots_on_target_10m | 0.5 |  |
| corners_10m | 0.5 |  |
| live_odd_over | 0.5 |  |
| live_xg | 0.5 |  |
| red_cards | 0.5 |  |
| possession | 0.5 |  |
| away_score | 0.47958846528019966 |  |

## 5. Conclusao

Nenhuma feature isolada excedeu o limiar de AUC (0.9). Sem indicios de leakage evidente.
