"""
Adaptador de inputs pré-jogo para o modelo Dixon-Coles já existente
(`src/engine/dixon_coles.py`).

Contexto (ver docs/AUDIT_MATEMATICA.md, secção 3.2): `dixon_coles_simulate_match()`
recebe `lambda_home`/`mu_away` como argumentos externos — o módulo Dixon-Coles em
si não calcula golos esperados a partir de dados, por desenho. Até agora, nenhum
caminho de produção fornecia esses valores, pelo que o modelo nunca era chamado
fora de testes.

Este módulo NÃO é um novo modelo de previsão de golos, nem reimplementa nada do
Dixon-Coles (`tau`, `dixon_coles_simulate_match`, `calculate_fractional_kelly`
permanecem exatamente como estão). É apenas o adaptador mínimo que faltava:
reparte a média de golos por jogo já observada no histórico de confrontos diretos
(`head_to_head.avg_total_goals`, o mesmo campo já consumido por
`src/model/predictor.py`) entre as duas equipas, com uma ligeira inclinação a
favor de quem tem melhor registo direto (`home_win_rate` vs `away_win_rate`) e a
favor da equipa da casa — na mesma ordem de grandeza do "+3" fixo de vantagem de
casa já usado em `predict_probability()`. Não há estimação de ataque/defesa por
MLE, nem decaimento temporal, nem qualquer otimização estatística: isso está
fora do âmbito pedido (não otimizar o próprio modelo, apenas colocá-lo em
produção).
"""

# Golos totais médios assumidos quando não há histórico de confrontos diretos
# (h2h ausente ou "avg_total_goals" inválido) — aproximação neutra de liga.
DEFAULT_AVG_TOTAL_GOALS = 2.5

# Vantagem de jogar em casa, expressa como fatia adicional da média de golos
# atribuída à equipa da casa. Mesma ordem de grandeza do "+3" fixo (numa escala
# de probabilidade 0-100) já usado em src/model/predictor.py.
HOME_ADVANTAGE_SHARE = 0.06

# Amplitude máxima do ajuste pela diferença de win-rate H2H (home_win_rate -
# away_win_rate, ambos em 0-100) sobre a repartição da média de golos.
MAX_STRENGTH_TILT = 0.15

# Nunca deixar lambda_home/mu_away chegarem a zero (ou perto disso) ao
# Dixon-Coles — um Poisson com lambda~0 degenera em "0 golos" quase certo.
MIN_LAMBDA = 0.35


def estimate_pregame_lambdas(h2h):
    """
    Deriva (lambda_home, mu_away) a partir dos dados de confrontos diretos
    (head-to-head) já devolvidos pela API, para alimentar
    `dixon_coles_simulate_match()` / `evaluate_match_value()`
    (`src/engine/dixon_coles.py`, `src/engine/value.py`).

    h2h:
        dict com (potencialmente) "avg_total_goals", "home_win_rate",
        "away_win_rate" — o mesmo formato já usado por
        `src/model/predictor.py::predict_probability()`. Pode ser None ou {}
        quando a API não tem histórico indexado para o confronto.

    Devolve sempre um par de floats > 0 (nunca lança exceção nem propaga
    None) — a pipeline não deve falhar por ausência de histórico H2H.
    """

    h2h = h2h or {}

    avg_total_goals = h2h.get("avg_total_goals")
    if not avg_total_goals or avg_total_goals <= 0:
        avg_total_goals = DEFAULT_AVG_TOTAL_GOALS

    home_rate = h2h.get("home_win_rate") or 0
    away_rate = h2h.get("away_win_rate") or 0

    rate_diff = home_rate - away_rate
    strength_tilt = max(
        -MAX_STRENGTH_TILT,
        min(MAX_STRENGTH_TILT, rate_diff / 200.0)
    )

    home_share = 0.5 + HOME_ADVANTAGE_SHARE + strength_tilt
    home_share = max(0.30, min(0.70, home_share))

    lambda_home = max(MIN_LAMBDA, round(avg_total_goals * home_share, 3))
    mu_away = max(MIN_LAMBDA, round(avg_total_goals * (1 - home_share), 3))

    return lambda_home, mu_away
