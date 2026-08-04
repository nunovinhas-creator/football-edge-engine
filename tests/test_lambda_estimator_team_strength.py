"""
Testes do Nível 0 da cascata de `lambda_estimator.py` (Melhoria #5 da
auditoria matemática): substituir o prior fixo de liga por forças de
ataque/defesa por equipa quando disponíveis em `head_to_head` (chaves
`team_strength_home_goals`/`team_strength_away_goals`/
`team_strength_sample_size`, preenchidas por
`src.historical_dataset.backtest_bridge.derive_h2h` a partir do Historical
Dataset Builder — nunca calculadas por `lambda_estimator.py`).

Cobrem:
  - `_resolve_dynamic_prior` isoladamente (fallback para o prior fixo,
    uso da força por equipa, invalidação de valores inválidos/negativos);
  - compatibilidade total com o comportamento anterior a esta melhoria
    (nenhuma das chaves novas presentes -> resultado idêntico ao prior
    fixo de sempre);
  - Nível 2 (encolhimento entre H2H e força por equipa) de ponta a ponta
    via `estimate_lambda_detailed`;
  - Nível 3 (prior de liga) só quando a força por equipa também está
    ausente/insuficiente;
  - estabilidade quando a força por equipa vem de poucas observações.
"""

import unittest

from src.engine.lambda_estimator import (
    LEAGUE_PRIOR_AWAY_GOALS,
    LEAGUE_PRIOR_HOME_GOALS,
    MAX_LAMBDA,
    MIN_LAMBDA,
    SHRINKAGE_K,
    _resolve_dynamic_prior,
    _shrink_to_prior,
    estimate_lambda,
    estimate_lambda_detailed,
)


class TestResolveDynamicPrior(unittest.TestCase):

    def test_no_team_strength_keys_returns_fixed_league_prior(self):
        self.assertEqual(
            _resolve_dynamic_prior({}),
            (LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS),
        )

    def test_partial_team_strength_keys_falls_back_to_fixed_prior(self):
        # só uma das duas equipas tem força calculável -> não há base para
        # a combinação; mantém o comportamento antigo (prior fixo), não
        # inventa metade de um prior.
        h2h = {"team_strength_home_goals": 2.0}
        self.assertEqual(
            _resolve_dynamic_prior(h2h),
            (LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS),
        )

    def test_negative_team_strength_is_rejected(self):
        h2h = {
            "team_strength_home_goals": -1.0,
            "team_strength_away_goals": 1.0,
            "team_strength_sample_size": 10.0,
        }
        self.assertEqual(
            _resolve_dynamic_prior(h2h),
            (LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS),
        )

    def test_garbage_type_is_rejected_without_raising(self):
        h2h = {
            "team_strength_home_goals": "oops",
            "team_strength_away_goals": 1.0,
            "team_strength_sample_size": 10.0,
        }
        self.assertEqual(
            _resolve_dynamic_prior(h2h),
            (LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS),
        )

    def test_large_team_strength_sample_dominates_the_fixed_prior(self):
        h2h = {
            "team_strength_home_goals": 3.0,
            "team_strength_away_goals": 0.5,
            "team_strength_sample_size": 10_000.0,
        }
        prior_home, prior_away = _resolve_dynamic_prior(h2h)
        self.assertAlmostEqual(prior_home, 3.0, places=2)
        self.assertAlmostEqual(prior_away, 0.5, places=2)

    def test_small_team_strength_sample_is_pulled_towards_fixed_prior(self):
        # reutiliza exatamente `_shrink_to_prior` (mesma função testada em
        # tests/test_lambda_estimator.py) -- não uma fórmula nova.
        h2h = {
            "team_strength_home_goals": 9.0,
            "team_strength_away_goals": 0.0,
            "team_strength_sample_size": 1.0,
        }
        prior_home, prior_away = _resolve_dynamic_prior(h2h)
        expected_home = _shrink_to_prior(9.0, 1.0, LEAGUE_PRIOR_HOME_GOALS, k=SHRINKAGE_K)
        expected_away = _shrink_to_prior(0.0, 1.0, LEAGUE_PRIOR_AWAY_GOALS, k=SHRINKAGE_K)
        self.assertAlmostEqual(prior_home, expected_home)
        self.assertAlmostEqual(prior_away, expected_away)
        # não pode ficar tão extremo quanto o valor bruto (9-0) com só 1 jogo.
        self.assertLess(prior_home, 9.0)
        self.assertGreater(prior_home, LEAGUE_PRIOR_HOME_GOALS)

    def test_missing_sample_size_defaults_to_zero_ie_pure_fixed_prior(self):
        h2h = {"team_strength_home_goals": 5.0, "team_strength_away_goals": 5.0}
        self.assertEqual(
            _resolve_dynamic_prior(h2h),
            (LEAGUE_PRIOR_HOME_GOALS, LEAGUE_PRIOR_AWAY_GOALS),
        )


class TestBackwardCompatibility(unittest.TestCase):
    """
    Compatibilidade total com o código existente: nenhum h2h usado pelos
    testes/pipeline anteriores a esta melhoria carrega as chaves
    `team_strength_*` -- o resultado tem de ser byte-a-byte idêntico ao
    comportamento anterior.
    """

    def test_none_h2h_unaffected(self):
        estimate = estimate_lambda_detailed(None)
        self.assertAlmostEqual(estimate.lambda_home, LEAGUE_PRIOR_HOME_GOALS, places=2)
        self.assertAlmostEqual(estimate.mu_away, LEAGUE_PRIOR_AWAY_GOALS, places=2)

    def test_h2h_without_team_strength_keys_matches_previous_behavior(self):
        h2h = {"total_matches": 6, "home_goals": 12, "away_goals": 6}
        estimate = estimate_lambda_detailed(h2h)
        raw_home_avg, raw_away_avg = 2.0, 1.0
        expected_home = _shrink_to_prior(raw_home_avg, 6, LEAGUE_PRIOR_HOME_GOALS, k=SHRINKAGE_K)
        expected_away = _shrink_to_prior(raw_away_avg, 6, LEAGUE_PRIOR_AWAY_GOALS, k=SHRINKAGE_K)
        self.assertAlmostEqual(estimate.lambda_home, round(expected_home, 3))
        self.assertAlmostEqual(estimate.mu_away, round(expected_away, 3))

    def test_public_contract_signature_unchanged(self):
        # estimate_lambda continua a aceitar só `h2h` e a devolver um par
        # de floats -- nenhum argumento novo foi introduzido.
        result = estimate_lambda({"avg_total_goals": 3.0})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestLevel2ShrinkageBetweenH2HAndTeamStrength(unittest.TestCase):
    """Nível 2: encolhimento do H2H observado para a força por equipa (Nível 0), não para o prior fixo."""

    def test_h2h_shrinks_towards_team_strength_not_fixed_prior_when_both_available(self):
        # Força por equipa (grande amostra) aponta para um jogo com muitos
        # golos; H2H (amostra pequena) sugere pouco. O resultado final deve
        # ficar mais perto da força por equipa do que ficaria do prior fixo
        # de liga.
        h2h = {
            "total_matches": 1,
            "home_goals": 1,
            "away_goals": 1,
            "team_strength_home_goals": 3.0,
            "team_strength_away_goals": 0.4,
            "team_strength_sample_size": 500.0,
        }
        estimate = estimate_lambda_detailed(h2h)

        without_team_strength = estimate_lambda_detailed(
            {"total_matches": 1, "home_goals": 1, "away_goals": 1}
        )

        self.assertGreater(estimate.lambda_home, without_team_strength.lambda_home)
        self.assertLess(estimate.mu_away, without_team_strength.mu_away)

    def test_large_h2h_sample_still_dominates_team_strength(self):
        # Nível 1 continua a poder dominar quando a amostra de H2H é
        # grande, exatamente como acontecia antes desta melhoria com o
        # prior fixo -- só o ALVO do encolhimento mudou, não a mecânica.
        h2h = {
            "total_matches": 10_000,
            "home_goals": 30_000,  # média 3.0
            "away_goals": 10_000,  # média 1.0
            "team_strength_home_goals": 0.5,
            "team_strength_away_goals": 3.0,
            "team_strength_sample_size": 1.0,
        }
        estimate = estimate_lambda_detailed(h2h)
        self.assertAlmostEqual(estimate.lambda_home, 3.0, places=1)
        self.assertAlmostEqual(estimate.mu_away, 1.0, places=1)


class TestLevel3FixedPriorOnlyWhenInsufficientInformation(unittest.TestCase):
    """Nível 3: o prior fixo de liga só é usado quando NEM H2H NEM força por equipa estão disponíveis."""

    def test_no_h2h_no_team_strength_uses_fixed_prior(self):
        estimate = estimate_lambda_detailed({})
        self.assertAlmostEqual(estimate.lambda_home, LEAGUE_PRIOR_HOME_GOALS, places=2)
        self.assertAlmostEqual(estimate.mu_away, LEAGUE_PRIOR_AWAY_GOALS, places=2)

    def test_no_h2h_but_team_strength_available_uses_team_strength_instead_of_fixed_prior(self):
        # Este é exatamente o cenário que a Melhoria #5 pretende resolver:
        # sem confrontos diretos, o estimador já não cai cegamente no prior
        # fixo se houver força por equipa disponível.
        h2h = {
            "team_strength_home_goals": 3.2,
            "team_strength_away_goals": 0.6,
            "team_strength_sample_size": 20.0,
        }
        estimate = estimate_lambda_detailed(h2h)
        self.assertNotAlmostEqual(estimate.lambda_home, LEAGUE_PRIOR_HOME_GOALS, places=2)
        self.assertGreater(estimate.lambda_home, LEAGUE_PRIOR_HOME_GOALS)
        self.assertLess(estimate.mu_away, LEAGUE_PRIOR_AWAY_GOALS)


class TestStabilityAndBounds(unittest.TestCase):

    def test_team_strength_from_single_observation_never_breaks_bounds(self):
        h2h = {
            "team_strength_home_goals": 50.0,
            "team_strength_away_goals": 0.0,
            "team_strength_sample_size": 1.0,
        }
        lambda_home, mu_away = estimate_lambda(h2h)
        self.assertGreaterEqual(lambda_home, MIN_LAMBDA)
        self.assertGreaterEqual(mu_away, MIN_LAMBDA)
        self.assertLessEqual(lambda_home, MAX_LAMBDA)
        self.assertLessEqual(mu_away, MAX_LAMBDA)

    def test_never_raises_with_malformed_team_strength_fields(self):
        cases = [
            {"team_strength_home_goals": None, "team_strength_away_goals": None},
            {"team_strength_home_goals": float("nan"), "team_strength_away_goals": 1.0},
            {"team_strength_sample_size": "not-a-number", "team_strength_home_goals": 1.0,
             "team_strength_away_goals": 1.0},
        ]
        for h2h in cases:
            lambda_home, mu_away = estimate_lambda(h2h)
            self.assertGreater(lambda_home, 0)
            self.assertGreater(mu_away, 0)


if __name__ == "__main__":
    unittest.main()
