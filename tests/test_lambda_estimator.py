"""
Testes do novo estimador pré-jogo de lambda_home/mu_away
(`src/engine/lambda_estimator.py`) — ver docs/05_lambda_estimator.md.

Cobrem:
  - as funções puras de cada "nível" de dados (recent_matches -> H2H goal
    totals -> avg_total_goals/prior), isoladamente;
  - a função de encolhimento estatístico (`_shrink_to_prior`) como unidade
    matemática independente;
  - `estimate_lambda_detailed()`/`estimate_lambda()` de ponta a ponta,
    incluindo todos os casos-limite já cobertos para o heurístico antigo
    (None, {}, avg_total_goals=0, amostras extremas) — nunca lança exceção,
    nunca devolve <= 0;
  - que o heurístico antigo (`pregame_lambda.py::estimate_pregame_lambdas`)
    continua exatamente inalterado (regressão de não-interferência);
  - integração de ponta a ponta com o Dixon-Coles já existente
    (`dixon_coles_simulate_match`), sem alterar essa função.
"""

import unittest

import numpy as np

from src.engine.dixon_coles import dixon_coles_simulate_match
from src.engine.pregame_lambda import (
    DEFAULT_AVG_TOTAL_GOALS,
    MIN_LAMBDA,
    estimate_pregame_lambdas,
)
from src.engine.lambda_estimator import (
    LEAGUE_PRIOR_HOME_GOALS,
    LEAGUE_PRIOR_AWAY_GOALS,
    MAX_LAMBDA,
    RECENT_MATCH_DECAY_RATE,
    _exponential_decay_effective_n,
    _shrink_to_prior,
    _split_from_recent_matches,
    _split_from_h2h_goal_totals,
    _split_from_avg_total_goals_or_prior,
    estimate_lambda,
    estimate_lambda_detailed,
)


class TestShrinkToPrior(unittest.TestCase):
    """Unidade matemática pura de encolhimento (empirical-Bayes shrinkage)."""

    def test_zero_sample_size_returns_pure_prior(self):
        self.assertAlmostEqual(_shrink_to_prior(9.0, 0.0, prior=2.0, k=4.0), 2.0)

    def test_large_sample_size_approaches_raw_value(self):
        shrunk = _shrink_to_prior(3.0, sample_size=10_000.0, prior=1.0, k=4.0)
        self.assertAlmostEqual(shrunk, 3.0, places=2)

    def test_sample_size_equal_to_k_averages_prior_and_raw(self):
        shrunk = _shrink_to_prior(4.0, sample_size=4.0, prior=2.0, k=4.0)
        self.assertAlmostEqual(shrunk, 3.0)  # média simples

    def test_negative_sample_size_is_clamped_to_zero(self):
        # nunca deve inflacionar o prior nem inverter o sinal
        shrunk = _shrink_to_prior(9.0, sample_size=-5.0, prior=2.0, k=4.0)
        self.assertAlmostEqual(shrunk, 2.0)

    def test_monotonic_between_prior_and_raw(self):
        prior, raw = 2.0, 5.0
        prev = prior
        for n in (0, 1, 2, 4, 8, 16, 100_000):
            shrunk = _shrink_to_prior(raw, n, prior=prior, k=4.0)
            self.assertGreaterEqual(shrunk, prev - 1e-9)
            prev = shrunk
        self.assertAlmostEqual(prev, raw, places=2)


class TestExponentialDecayEffectiveN(unittest.TestCase):
    """
    Amostra efetiva (design effect) de uma média ponderada por decaimento
    exponencial — corrige um problema real encontrado no benchmark
    (`scripts/benchmark_lambda_estimator.py`, secção 2): sem isto, o Nível A
    (`recent_matches`) tratava `n` jogos fortemente ponderados para os mais
    recentes como se fossem `n` observações igualmente independentes,
    sub-encolhendo a estimativa e piorando o MSE/Brier em amostras grandes
    face ao heurístico antigo — o oposto do objetivo desta tarefa.
    """

    def test_n_eff_equals_n_for_single_observation(self):
        self.assertAlmostEqual(_exponential_decay_effective_n(1, 0.35), 1.0)

    def test_n_eff_is_never_larger_than_n(self):
        for n in (1, 2, 5, 10, 30, 100):
            self.assertLessEqual(_exponential_decay_effective_n(n, RECENT_MATCH_DECAY_RATE), n)

    def test_n_eff_saturates_instead_of_growing_unboundedly(self):
        # com pesos exponenciais, n_eff estabiliza — não cresce linearmente
        # com n, ao contrário de uma média simples (onde n_eff == n).
        n_eff_20 = _exponential_decay_effective_n(20, RECENT_MATCH_DECAY_RATE)
        n_eff_200 = _exponential_decay_effective_n(200, RECENT_MATCH_DECAY_RATE)
        self.assertLess(n_eff_200 - n_eff_20, 1.0)

    def test_zero_decay_rate_keeps_full_sample_size(self):
        # decay_rate=0 -> pesos uniformes -> n_eff == n (caso degenerado)
        self.assertAlmostEqual(_exponential_decay_effective_n(10, 0.0), 10.0, places=4)

    def test_zero_observations_has_zero_effective_size(self):
        self.assertEqual(_exponential_decay_effective_n(0, RECENT_MATCH_DECAY_RATE), 0.0)


class TestSplitFromRecentMatches(unittest.TestCase):
    """Nível A: repartição ponderada por recência a partir de recent_matches."""

    def test_none_or_missing_returns_none(self):
        self.assertIsNone(_split_from_recent_matches({}))
        self.assertIsNone(_split_from_recent_matches({"recent_matches": None}))

    def test_not_a_list_returns_none(self):
        self.assertIsNone(_split_from_recent_matches({"recent_matches": "oops"}))

    def test_too_few_usable_entries_returns_none(self):
        h2h = {"recent_matches": [{"home_goals": 2, "away_goals": 1}]}
        self.assertIsNone(_split_from_recent_matches(h2h))

    def test_malformed_entries_are_skipped(self):
        h2h = {
            "recent_matches": [
                {"home_goals": 2, "away_goals": 1},
                {"home_goals": None, "away_goals": 1},  # descartado
                "not-a-dict",  # descartado
                {"home_goals": 3, "away_goals": 0},
            ]
        }
        result = _split_from_recent_matches(h2h)
        self.assertIsNotNone(result)
        home_avg, away_avg, n = result
        # amostra efetiva (design effect) das 2 entradas válidas apenas —
        # as 2 descartadas não podem ter contribuído para o cálculo.
        self.assertAlmostEqual(n, _exponential_decay_effective_n(2, RECENT_MATCH_DECAY_RATE))

    def test_recent_goals_are_weighted_towards_the_end_of_the_list(self):
        # Sem "date": assume-se API-mais-recente-primeiro -> inverte-se para
        # cronológico (mais antigo primeiro) antes do decaimento. Por isso o
        # PRIMEIRO elemento da lista recebe o maior peso.
        h2h_recent_blowout_first = {
            "recent_matches": [
                {"home_goals": 5, "away_goals": 0},
                {"home_goals": 0, "away_goals": 0},
                {"home_goals": 0, "away_goals": 0},
            ]
        }
        h2h_recent_blowout_last = {
            "recent_matches": [
                {"home_goals": 0, "away_goals": 0},
                {"home_goals": 0, "away_goals": 0},
                {"home_goals": 5, "away_goals": 0},
            ]
        }
        home_avg_first, _, _ = _split_from_recent_matches(h2h_recent_blowout_first)
        home_avg_last, _, _ = _split_from_recent_matches(h2h_recent_blowout_last)
        self.assertGreater(home_avg_first, home_avg_last)

    def test_uses_date_ordering_when_available(self):
        h2h = {
            "recent_matches": [
                {"date": "2020-01-01", "home_goals": 0, "away_goals": 0},
                {"date": "2026-01-01", "home_goals": 5, "away_goals": 0},
                {"date": "2010-01-01", "home_goals": 0, "away_goals": 0},
            ]
        }
        home_avg, _, n = _split_from_recent_matches(h2h)
        self.assertAlmostEqual(n, _exponential_decay_effective_n(3, RECENT_MATCH_DECAY_RATE))
        # jogo mais recente por data (5-0) domina apesar de estar no meio da lista
        self.assertGreater(home_avg, 1.0)


class TestSplitFromH2HGoalTotals(unittest.TestCase):
    """Nível B: repartição empírica direta a partir de home_goals/away_goals."""

    def test_missing_total_matches_returns_none(self):
        self.assertIsNone(_split_from_h2h_goal_totals({"home_goals": 10, "away_goals": 5}))

    def test_missing_goal_fields_returns_none(self):
        self.assertIsNone(_split_from_h2h_goal_totals({"total_matches": 5}))

    def test_computes_per_match_average(self):
        h2h = {"total_matches": 5, "home_goals": 10, "away_goals": 5}
        home_avg, away_avg, n = _split_from_h2h_goal_totals(h2h)
        self.assertAlmostEqual(home_avg, 2.0)
        self.assertAlmostEqual(away_avg, 1.0)
        self.assertEqual(n, 5)

    def test_negative_goals_rejected(self):
        h2h = {"total_matches": 5, "home_goals": -1, "away_goals": 5}
        self.assertIsNone(_split_from_h2h_goal_totals(h2h))


class TestSplitFromAvgTotalGoalsOrPrior(unittest.TestCase):
    """Nível C/D: delega ao heurístico existente (sem duplicar a lógica)."""

    def test_delegates_to_legacy_heuristic(self):
        h2h = {"avg_total_goals": 4.0, "home_win_rate": 60, "away_win_rate": 20}
        legacy_home, legacy_away = estimate_pregame_lambdas(h2h)

        home_avg, away_avg, _ = _split_from_avg_total_goals_or_prior(h2h)

        self.assertAlmostEqual(home_avg, legacy_home)
        self.assertAlmostEqual(away_avg, legacy_away)

    def test_sample_size_discounted_from_total_matches(self):
        h2h = {"avg_total_goals": 3.0, "total_matches": 10}
        _, _, sample_size = _split_from_avg_total_goals_or_prior(h2h)
        self.assertAlmostEqual(sample_size, 5.0)  # 10 * TIER_C_CONFIDENCE_DISCOUNT(0.5)

    def test_no_h2h_at_all_has_zero_sample_size(self):
        _, _, sample_size = _split_from_avg_total_goals_or_prior({})
        self.assertEqual(sample_size, 0.0)


class TestEstimateLambdaDetailed(unittest.TestCase):
    """Comportamento de ponta a ponta, incluindo escolha do "nível" certo."""

    def test_none_h2h_never_raises_and_uses_prior(self):
        estimate = estimate_lambda_detailed(None)
        self.assertEqual(estimate.tier, "avg_total_goals_or_prior")
        self.assertAlmostEqual(estimate.lambda_home, LEAGUE_PRIOR_HOME_GOALS, places=2)
        self.assertAlmostEqual(estimate.mu_away, LEAGUE_PRIOR_AWAY_GOALS, places=2)

    def test_empty_dict_behaves_like_none(self):
        self.assertEqual(estimate_lambda_detailed({}), estimate_lambda_detailed(None))

    def test_garbage_input_types_never_raise(self):
        for garbage in ["not-a-dict", 42, 3.14, ["list"], object()]:
            estimate = estimate_lambda_detailed(garbage)
            self.assertGreater(estimate.lambda_home, 0)
            self.assertGreater(estimate.mu_away, 0)

    def test_prefers_recent_matches_over_h2h_totals(self):
        h2h = {
            "total_matches": 20,
            "home_goals": 20,  # média agregada: 1.0
            "away_goals": 20,  # média agregada: 1.0
            "recent_matches": [
                {"home_goals": 4, "away_goals": 0},
                {"home_goals": 4, "away_goals": 0},
                {"home_goals": 4, "away_goals": 0},
            ],
        }
        estimate = estimate_lambda_detailed(h2h)
        self.assertEqual(estimate.tier, "recent_matches")
        # forma recente muito mais forte para a casa do que a média agregada sugeria
        self.assertGreater(estimate.lambda_home, estimate.mu_away)

    def test_prefers_h2h_totals_over_avg_total_goals_when_no_recent_matches(self):
        h2h = {"total_matches": 10, "home_goals": 25, "away_goals": 5, "avg_total_goals": 1.0}
        estimate = estimate_lambda_detailed(h2h)
        self.assertEqual(estimate.tier, "h2h_goal_totals")

    def test_falls_back_to_avg_total_goals_tier(self):
        estimate = estimate_lambda_detailed({"avg_total_goals": 4.0})
        self.assertEqual(estimate.tier, "avg_total_goals_or_prior")

    def test_small_sample_is_pulled_towards_league_prior(self):
        # 1 único jogo direto, vitória esmagadora da casa: a média bruta
        # (5-0) não deve sobreviver ao encolhimento com uma amostra tão
        # pequena — deve ficar claramente mais perto do prior do que do
        # resultado bruto de um só jogo.
        h2h = {"total_matches": 1, "home_goals": 5, "away_goals": 0}
        estimate = estimate_lambda_detailed(h2h)
        self.assertLess(estimate.lambda_home, 5.0)
        self.assertGreater(estimate.lambda_home, LEAGUE_PRIOR_HOME_GOALS)

    def test_large_sample_trusts_the_data_more_than_a_small_one(self):
        small = estimate_lambda_detailed({"total_matches": 2, "home_goals": 6, "away_goals": 0})
        large = estimate_lambda_detailed({"total_matches": 40, "home_goals": 120, "away_goals": 0})
        # ambos os casos "empiricamente" apontam para o mesmo home_avg bruto
        # (3.0), mas a amostra maior deve ficar mais perto de 3.0
        self.assertGreater(large.lambda_home, small.lambda_home)

    def test_never_returns_lambda_outside_bounds(self):
        cases = [
            None,
            {},
            {"avg_total_goals": 0},
            {"avg_total_goals": -5},
            {"total_matches": 1, "home_goals": 50, "away_goals": 0},
            {"total_matches": 1000, "home_goals": 9000, "away_goals": 0},
        ]
        for h2h in cases:
            lambda_home, mu_away = estimate_lambda(h2h)
            self.assertGreaterEqual(lambda_home, MIN_LAMBDA)
            self.assertGreaterEqual(mu_away, MIN_LAMBDA)
            self.assertLessEqual(lambda_home, MAX_LAMBDA)
            self.assertLessEqual(mu_away, MAX_LAMBDA)

    def test_estimate_lambda_matches_detailed_tuple(self):
        h2h = {"total_matches": 6, "home_goals": 12, "away_goals": 6}
        self.assertEqual(estimate_lambda(h2h), estimate_lambda_detailed(h2h).as_tuple())


class TestLegacyHeuristicUnaffected(unittest.TestCase):
    """
    Regressão de não-interferência: `lambda_estimator.py` importa e reutiliza
    `pregame_lambda.py::estimate_pregame_lambdas`, mas não pode ter alterado
    o seu comportamento (usado como fallback defensivo em
    src/collector/client.py e por tests/test_dixon_coles_pipeline.py).
    """

    def test_legacy_still_preserves_avg_total_goals_exactly(self):
        lambda_home, mu_away = estimate_pregame_lambdas({"avg_total_goals": 4.0})
        self.assertAlmostEqual(lambda_home + mu_away, 4.0, places=2)

    def test_legacy_default_unchanged(self):
        lambda_home, mu_away = estimate_pregame_lambdas(None)
        self.assertAlmostEqual(lambda_home + mu_away, DEFAULT_AVG_TOTAL_GOALS, places=2)


class TestIntegrationWithDixonColes(unittest.TestCase):
    """
    De ponta a ponta: a saída do novo estimador continua a produzir uma
    matriz Dixon-Coles válida, sem qualquer alteração a
    `dixon_coles_simulate_match()`/`tau()`.
    """

    def test_output_feeds_a_valid_probability_matrix(self):
        h2h = {
            "total_matches": 12,
            "home_goals": 22,
            "away_goals": 14,
            "home_win_rate": 55,
            "away_win_rate": 25,
        }
        lambda_home, mu_away = estimate_lambda(h2h)
        matrix = dixon_coles_simulate_match(lambda_home, mu_away)

        self.assertAlmostEqual(float(np.sum(matrix)), 1.0, places=9)
        self.assertTrue(np.all(matrix >= 0))

    def test_stronger_recent_home_form_increases_home_win_probability(self):
        from src.engine.value import estimate_pregame_probabilities

        weak_home_h2h = {
            "recent_matches": [
                {"home_goals": 0, "away_goals": 2},
                {"home_goals": 0, "away_goals": 2},
                {"home_goals": 0, "away_goals": 2},
            ]
        }
        strong_home_h2h = {
            "recent_matches": [
                {"home_goals": 3, "away_goals": 0},
                {"home_goals": 3, "away_goals": 0},
                {"home_goals": 3, "away_goals": 0},
            ]
        }

        weak_lambda_home, weak_mu_away = estimate_lambda(weak_home_h2h)
        strong_lambda_home, strong_mu_away = estimate_lambda(strong_home_h2h)

        weak_probs = estimate_pregame_probabilities(weak_lambda_home, weak_mu_away)
        strong_probs = estimate_pregame_probabilities(strong_lambda_home, strong_mu_away)

        self.assertGreater(strong_probs["home"], weak_probs["home"])


if __name__ == "__main__":
    unittest.main()
