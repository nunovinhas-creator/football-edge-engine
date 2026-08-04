# Relatorio de Validacao Cruzada (Group K-Fold por match_id)

Metodo de validacao cruzada: `StratifiedGroupKFold` com 5 folds, agrupados por `match_id` (nenhum jogo aparece simultaneamente em treino e teste em nenhum fold).

- Total de jogos no dataset: 96
- Total de snapshots no dataset: 2013

## Estatisticas por fold

| Fold | Jogos treino | Jogos teste | Snapshots treino | Snapshots teste | Overlap match_id | Teste com 1 classe | Treino com 1 classe |
|---|---|---|---|---|---|---|---|
| 1 | 77 | 19 | 1615 | 398 | 0 | Nao | Nao |
| 2 | 76 | 20 | 1616 | 397 | 0 | Nao | Nao |
| 3 | 74 | 22 | 1617 | 396 | 0 | Nao | Nao |
| 4 | 78 | 18 | 1590 | 423 | 0 | Nao | Nao |
| 5 | 79 | 17 | 1614 | 399 | 0 | Nao | Nao |

## Metricas medias por modelo (media +/- desvio-padrao entre folds validos)

| Modelo | Folds validos | Folds ignorados | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| LightGBM (melhor modelo) | 5 | 0 | 0.5372 +/- 0.0261 | 0.3645 +/- 0.0216 | 0.6511 +/- 0.0586 | 0.4666 +/- 0.0287 | 0.6095 +/- 0.0266 (n=5) |
| HistGradientBoosting | 5 | 0 | 0.6722 +/- 0.0185 | 0.4040 +/- 0.0619 | 0.1119 +/- 0.0459 | 0.1718 +/- 0.0582 | 0.6007 +/- 0.0249 (n=5) |
| ExtraTrees | 5 | 0 | 0.5108 +/- 0.0191 | 0.3579 +/- 0.0196 | 0.7177 +/- 0.0398 | 0.4774 +/- 0.0239 | 0.6006 +/- 0.0367 (n=5) |
| BalancedRandomForest | 5 | 0 | 0.4900 +/- 0.0194 | 0.3663 +/- 0.0183 | 0.8738 +/- 0.0577 | 0.5160 +/- 0.0254 | 0.5980 +/- 0.0372 (n=5) |
| RandomForest | 5 | 0 | 0.4864 +/- 0.0109 | 0.3650 +/- 0.0145 | 0.8785 +/- 0.0442 | 0.5157 +/- 0.0210 | 0.5968 +/- 0.0288 (n=5) |
| GradientBoosting | 5 | 0 | 0.6575 +/- 0.0146 | 0.3815 +/- 0.0575 | 0.1693 +/- 0.0534 | 0.2313 +/- 0.0600 | 0.5947 +/- 0.0276 (n=5) |

## Melhor modelo

`LightGBM` — selecionado por ROC-AUC media = 0.6095

## Metricas por fold (detalhe)

### RandomForest

| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | Sim | 0.4698 | 0.3618 | 0.8661 | 0.5104 | 0.5436 |  |
| 2 | Sim | 0.5013 | 0.3797 | 0.8819 | 0.5308 | 0.6203 |  |
| 3 | Sim | 0.4874 | 0.3621 | 0.8537 | 0.5085 | 0.5895 |  |
| 4 | Sim | 0.4799 | 0.3411 | 0.8306 | 0.4836 | 0.6171 |  |
| 5 | Sim | 0.4937 | 0.3805 | 0.9603 | 0.5450 | 0.6137 |  |

### ExtraTrees

| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | Sim | 0.4925 | 0.3494 | 0.6850 | 0.4628 | 0.5384 |  |
| 2 | Sim | 0.5466 | 0.3882 | 0.7244 | 0.5055 | 0.6130 |  |
| 3 | Sim | 0.5126 | 0.3622 | 0.7480 | 0.4881 | 0.5949 |  |
| 4 | Sim | 0.5035 | 0.3280 | 0.6613 | 0.4385 | 0.6045 |  |
| 5 | Sim | 0.4987 | 0.3619 | 0.7698 | 0.4924 | 0.6521 |  |

### BalancedRandomForest

| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | Sim | 0.4673 | 0.3597 | 0.8583 | 0.5070 | 0.5264 |  |
| 2 | Sim | 0.5239 | 0.3885 | 0.8504 | 0.5333 | 0.6250 |  |
| 3 | Sim | 0.4848 | 0.3618 | 0.8618 | 0.5096 | 0.5967 |  |
| 4 | Sim | 0.4775 | 0.3378 | 0.8145 | 0.4775 | 0.6183 |  |
| 5 | Sim | 0.4962 | 0.3839 | 0.9841 | 0.5523 | 0.6235 |  |

### GradientBoosting

| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | Sim | 0.6508 | 0.4032 | 0.1969 | 0.2646 | 0.5416 |  |
| 2 | Sim | 0.6700 | 0.4545 | 0.1575 | 0.2339 | 0.6125 |  |
| 3 | Sim | 0.6389 | 0.3387 | 0.1707 | 0.2270 | 0.5982 |  |
| 4 | Sim | 0.6785 | 0.4167 | 0.2419 | 0.3061 | 0.6195 |  |
| 5 | Sim | 0.6491 | 0.2941 | 0.0794 | 0.1250 | 0.6019 |  |

### LightGBM

| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | Sim | 0.4950 | 0.3432 | 0.6378 | 0.4463 | 0.5691 |  |
| 2 | Sim | 0.5466 | 0.3822 | 0.6772 | 0.4886 | 0.6194 |  |
| 3 | Sim | 0.5707 | 0.3956 | 0.7236 | 0.5115 | 0.6379 |  |
| 4 | Sim | 0.5225 | 0.3402 | 0.6694 | 0.4511 | 0.6330 |  |
| 5 | Sim | 0.5514 | 0.3613 | 0.5476 | 0.4353 | 0.5880 |  |

### HistGradientBoosting

| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | Sim | 0.6482 | 0.3556 | 0.1260 | 0.1860 | 0.5562 |  |
| 2 | Sim | 0.6599 | 0.3333 | 0.0630 | 0.1060 | 0.6274 |  |
| 3 | Sim | 0.6692 | 0.3750 | 0.0976 | 0.1548 | 0.6173 |  |
| 4 | Sim | 0.7021 | 0.4800 | 0.1935 | 0.2759 | 0.6091 |  |
| 5 | Sim | 0.6817 | 0.4762 | 0.0794 | 0.1361 | 0.5935 |  |
