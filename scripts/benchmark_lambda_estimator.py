"""
Benchmark: estimador de lambda antigo (heurístico, `src/engine/pregame_lambda.py`)
vs. novo (`src/engine/lambda_estimator.py`).

Ver `docs/05_lambda_estimator.md`, secção "Validation", para a leitura
completa dos resultados. Este script não altera nem re-implementa
nenhuma fórmula do motor — Dixon-Coles, Kelly, edge/EV — apenas reutiliza
o que já existe em `src.engine.*` e `src.backtest.historical.statistics`
(Brier Score / Log Loss, para não duplicar essas fórmulas).

Duas secções, deliberadamente distintas:

1. "Scenario comparison" — determinístico, sobre inputs de H2H realistas
   (não é uma medição de qualidade preditiva, só mostra o que cada
   estimador produz para o mesmo input).

2. "Synthetic recovery simulation" — um estudo de validação da MECÂNICA
   ESTATÍSTICA do estimador (encolhimento/shrinkage), usando dados
   INTEIRAMENTE SINTÉTICOS com uma verdade fundamental (`lambda` real)
   conhecida por construção. Isto NÃO é uma alegação de desempenho
   preditivo em jogos reais — é um teste de unidade estatístico alargado,
   claramente rotulado como tal. Ver `docs/05_lambda_estimator.md` para a
   explicação de por que motivo um backtest real (Brier/log-loss/ROI sobre
   jogos e odds reais) não pôde ser feito com os dados atualmente no
   repositório, e o que seria necessário para o fazer.

Execução: `python scripts/benchmark_lambda_estimator.py`
"""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.engine.pregame_lambda import estimate_pregame_lambdas
from src.engine.lambda_estimator import estimate_lambda, estimate_lambda_detailed
from src.engine.value import estimate_pregame_probabilities
from src.backtest.historical.statistics import brier_score, log_loss

RNG_SEED = 42  # reprodutibilidade — ver docs/AUDIT_MATEMATICA.md §12.2 (nº9)


@lru_cache(maxsize=None)
def _cached_home_win_prob(lambda_home: float, mu_away: float) -> float:
    """
    `dixon_coles_simulate_match()` (não alterado, `src/engine/dixon_coles.py`)
    chama `scipy.stats.poisson.pmf` célula a célula (81 células por matriz) —
    correto mas caro para chamar dezenas de milhares de vezes num script de
    benchmark. Como o encolhimento estatístico faz com que valores pequenos
    de `n` (amostra H2H) repitam frequentemente o mesmo (lambda_home,
    mu_away) já arredondado a 3 casas decimais, uma cache pura por par de
    valores elimina a maior parte do trabalho redundante sem tocar em
    `dixon_coles.py` nem alterar nenhum resultado.
    """
    return estimate_pregame_probabilities(lambda_home, mu_away)["home"]


# --------------------------------------------------------------------------
# Secção 1: comparação determinística de cenários
# --------------------------------------------------------------------------

SCENARIOS = [
    ("Sem histórico H2H", None),
    ("H2H mínimo (1 jogo, goleada da casa)", {"total_matches": 1, "home_goals": 5, "away_goals": 0}),
    (
        "H2H moderado (8 jogos, ligeiro favoritismo casa)",
        {"total_matches": 8, "avg_total_goals": 3.0, "home_win_rate": 60, "away_win_rate": 20},
    ),
    (
        "H2H robusto (30 jogos, dados por equipa)",
        {"total_matches": 30, "home_goals": 54, "away_goals": 33, "home_win_rate": 53, "away_win_rate": 27},
    ),
    (
        "H2H com jogos recentes (forma recente inverte tendência histórica)",
        {
            "total_matches": 12,
            "home_goals": 12,  # média histórica: empate técnico (1.0 vs 1.0)
            "away_goals": 12,
            "recent_matches": [
                {"date": "2024-01-01", "home_goals": 0, "away_goals": 2},
                {"date": "2024-06-01", "home_goals": 0, "away_goals": 3},
                {"date": "2025-01-01", "home_goals": 1, "away_goals": 3},
                {"date": "2025-08-01", "home_goals": 0, "away_goals": 2},
            ],
        },
    ),
]


def run_scenario_comparison():
    print("=" * 100)
    print("SECÇÃO 1 — Comparação de cenários (determinístico, sem verdade fundamental)")
    print("=" * 100)

    rows = []
    for name, h2h in SCENARIOS:
        old_home, old_away = estimate_pregame_lambdas(h2h)
        new_estimate = estimate_lambda_detailed(h2h)

        old_probs = estimate_pregame_probabilities(old_home, old_away)
        new_probs = estimate_pregame_probabilities(new_estimate.lambda_home, new_estimate.mu_away)

        rows.append(
            {
                "cenário": name,
                "λ_home (antigo)": old_home,
                "μ_away (antigo)": old_away,
                "P(home) antigo": round(old_probs["home"], 3),
                "λ_home (novo)": new_estimate.lambda_home,
                "μ_away (novo)": new_estimate.mu_away,
                "P(home) novo": round(new_probs["home"], 3),
                "nível usado (novo)": new_estimate.tier,
            }
        )

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 160, "display.max_colwidth", 45):
        print(df.to_string(index=False))
    print()
    return df


# --------------------------------------------------------------------------
# Secção 2: simulação de recuperação sintética (validação da mecânica)
# --------------------------------------------------------------------------

TRUE_LAMBDA_SCENARIOS = [
    ("equilibrado", 1.4, 1.2),
    ("casa dominante", 2.2, 0.8),
    ("fora dominante", 0.9, 1.9),
]
SAMPLE_SIZES = [1, 2, 3, 5, 10, 20, 50]
N_TRIALS_PER_SIZE = 400
N_OUT_OF_SAMPLE_MATCHES = 80


def _simulate_h2h_sample(rng, true_lambda_home, true_mu_away, n):
    home_goals = rng.poisson(true_lambda_home, size=n)
    away_goals = rng.poisson(true_mu_away, size=n)

    home_wins = int(np.sum(home_goals > away_goals))
    away_wins = int(np.sum(away_goals > home_goals))

    h2h = {
        "total_matches": n,
        "home_goals": int(np.sum(home_goals)),
        "away_goals": int(np.sum(away_goals)),
        "avg_total_goals": float(np.mean(home_goals + away_goals)),
        "home_win_rate": 100.0 * home_wins / n,
        "away_win_rate": 100.0 * away_wins / n,
        "recent_matches": [
            {"home_goals": int(h), "away_goals": int(a)}
            for h, a in zip(home_goals, away_goals)
        ],
    }
    return h2h


def _out_of_sample_outcomes(rng, true_lambda_home, true_mu_away, n_matches):
    """Jogos NOVOS (fora da amostra usada para estimar lambda), a partir da
    distribuição VERDADEIRA — devolve o outcome real "casa venceu" (0/1)."""
    home_goals = rng.poisson(true_lambda_home, size=n_matches)
    away_goals = rng.poisson(true_mu_away, size=n_matches)
    return (home_goals > away_goals).astype(int)


def run_synthetic_recovery_simulation():
    print("=" * 100)
    print("SECÇÃO 2 — Simulação de recuperação sintética")
    print("(validação da MECÂNICA do estimador — verdade fundamental conhecida por")
    print(" construção; NÃO é uma alegação de desempenho preditivo em jogos reais)")
    print("=" * 100)

    rng = np.random.default_rng(RNG_SEED)
    results = []

    for scenario_name, true_lh, true_ma in TRUE_LAMBDA_SCENARIOS:
        for n in SAMPLE_SIZES:
            old_sq_errors = []
            new_sq_errors = []
            # Acumula (probabilidade prevista, outcome real) de TODOS os
            # trials num único DataFrame por (cenário, n, estimador) — a
            # média de Brier/Log Loss por trial (todos com o mesmo nº de
            # jogos fora da amostra) é matematicamente idêntica a calcular
            # a métrica uma só vez sobre o conjunto combinado, e evita
            # milhares de construções de DataFrame desnecessárias.
            old_rows = {"probability": [], "won": []}
            new_rows = {"probability": [], "won": []}

            for _ in range(N_TRIALS_PER_SIZE):
                h2h = _simulate_h2h_sample(rng, true_lh, true_ma, n)

                old_lh, old_ma = estimate_pregame_lambdas(h2h)
                new_lh, new_ma = estimate_lambda(h2h)

                old_sq_errors.append((old_lh - true_lh) ** 2 + (old_ma - true_ma) ** 2)
                new_sq_errors.append((new_lh - true_lh) ** 2 + (new_ma - true_ma) ** 2)

                outcomes = _out_of_sample_outcomes(rng, true_lh, true_ma, N_OUT_OF_SAMPLE_MATCHES)

                old_prob_home = _cached_home_win_prob(old_lh, old_ma)
                new_prob_home = _cached_home_win_prob(new_lh, new_ma)

                old_rows["probability"].extend([old_prob_home] * N_OUT_OF_SAMPLE_MATCHES)
                old_rows["won"].extend(outcomes.tolist())
                new_rows["probability"].extend([new_prob_home] * N_OUT_OF_SAMPLE_MATCHES)
                new_rows["won"].extend(outcomes.tolist())

            old_df = pd.DataFrame(old_rows)
            new_df = pd.DataFrame(new_rows)

            results.append(
                {
                    "cenário verdadeiro": scenario_name,
                    "n (jogos H2H)": n,
                    "MSE λ (antigo)": round(float(np.mean(old_sq_errors)), 4),
                    "MSE λ (novo)": round(float(np.mean(new_sq_errors)), 4),
                    "Brier (antigo)": brier_score(old_df),
                    "Brier (novo)": brier_score(new_df),
                    "LogLoss (antigo)": log_loss(old_df),
                    "LogLoss (novo)": log_loss(new_df),
                }
            )

    df = pd.DataFrame(results)
    with pd.option_context("display.width", 160):
        print(df.to_string(index=False))
    print()

    print("-" * 100)
    print("Resumo agregado (média sobre todos os cenários/tamanhos de amostra):")
    print("-" * 100)
    summary = {
        "MSE λ (antigo)": df["MSE λ (antigo)"].mean(),
        "MSE λ (novo)": df["MSE λ (novo)"].mean(),
        "Brier (antigo)": df["Brier (antigo)"].mean(),
        "Brier (novo)": df["Brier (novo)"].mean(),
        "LogLoss (antigo)": df["LogLoss (antigo)"].mean(),
        "LogLoss (novo)": df["LogLoss (novo)"].mean(),
    }
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print()

    print("Leitura dos resultados observados nesta execução (ver docs/05_lambda_estimator.md,")
    print("secção 'Validation', para a tabela completa e discussão):")
    print("  - Para n pequeno (1-5 jogos H2H, o caso mais comum em confrontos diretos reais),")
    print("    o novo estimador reduz claramente o MSE de lambda e o Brier/Log Loss fora da")
    print("    amostra face ao heurístico antigo — efeito esperado do encolhimento (shrinkage)")
    print("    para o prior de liga em amostras pequenas.")
    print("  - Para n grande (>=20), o MSE do novo estimador estabiliza (não continua a cair)")
    print("    em vez de convergir para o do antigo. Isto é uma CONSEQUÊNCIA DELIBERADA do Nível")
    print("    A (recent_matches): a ponderação por recência tem uma amostra EFETIVA limitada")
    print("    (~5-6 jogos 'equivalentes', ver _exponential_decay_effective_n), porque o desenho")
    print("    escolhe responder a mudanças de forma recentes em vez de convergir para a média")
    print("    de longo prazo — um trade-off clássico enviesamento/variância, não um defeito.")
    print("    Ainda assim, o Brier/Log Loss (que é o que importa para P(home)) permanece igual")
    print("    ou melhor do que o antigo em quase todos os cenários/tamanhos de amostra testados.")
    print()
    return df


def run_missing_real_backtest_note():
    print("=" * 100)
    print("SECÇÃO 3 — Backtest real (Brier/Log Loss/Calibração/ROI sobre jogos reais)")
    print("=" * 100)
    print(
        "NÃO EXECUTADO — dados insuficientes no repositório. Ver "
        "docs/05_lambda_estimator.md, secção 'Validation' / 'Missing data "
        "requirements' para a explicação completa e o que seria necessário "
        "para o fazer (dataset ligando snapshots de head_to_head tal como "
        "estavam disponíveis pré-jogo, resultado final e odds reais)."
    )
    print()


if __name__ == "__main__":
    run_scenario_comparison()
    run_synthetic_recovery_simulation()
    run_missing_real_backtest_note()
