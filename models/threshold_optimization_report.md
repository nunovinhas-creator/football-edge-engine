# Relatorio de Otimizacao do Threshold de Decisao

Modelo vencedor: `LightGBM`

Metrica de otimizacao: `f1`
Threshold otimo: 0.2000
Valor da metrica no threshold otimo: 0.5309
Amostras de validacao (out-of-fold) usadas: 2013

## Comparacao: threshold 0.5 (default) vs threshold otimo

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 (default) | 0.3750 | 0.0144 | 0.0276 |
| 0.2000 (otimo) | 0.3651 | 0.9729 | 0.5309 |

## Matriz de confusao — threshold 0.5 (default)

| | Previsto: sem golo | Previsto: golo |
|---|---|---|
| Real: sem golo | TN=1371 | FP=15 |
| Real: golo | FN=618 | TP=9 |

## Matriz de confusao — threshold 0.2000 (otimo)

| | Previsto: sem golo | Previsto: golo |
|---|---|---|
| Real: sem golo | TN=325 | FP=1061 |
| Real: golo | FN=17 | TP=610 |
