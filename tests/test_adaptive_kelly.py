"""
Testes da Melhoria #6 (auditoria matemática, `docs/AUDIT_MATEMATICA.md` §7):
escala automática da fração de Kelly pela confiança do modelo, reutilizando
exclusivamente `LambdaEstimate.tier` / `LambdaEstimate.effective_sample_size`
(já produzidos por `src.engine.lambda_estimator.estimate_lambda_detailed`,
Melhoria #5).

Cobre:
    - `src.engine.kelly.calculate_confidence_multiplier` /
      `calculate_adaptive_kelly_fraction` (a função única).
    - Retrocompatibilidade de `fractional_kelly`,
      `src.engine.dixon_coles.calculate_fractional_kelly`,
      `src.engine.decision.DecisionEngine.evaluate_bet` e
      `src.backtest.historical.staking.KellyStake` quando os metadados de
      confiança não são fornecidos.
    - Regressão: nenhuma das quatro implementações mudou de valor para o
      caso sem metadados (o único caso que existia antes desta melhoria).
    - Continuidade e limites: a fração nunca escala para cima da fração
      base, e nunca "salta" com pequenas variações de `effective_sample_size`.

Vocabulário "tier HIGH/MEDIUM/LOW" usado nos nomes dos testes abaixo
mapeia diretamente para os três valores possíveis de `LambdaEstimate.tier`
("recent_matches" | "h2h_goal_totals" | "avg_total_goals_or_prior"), na
mesma ordem de qualidade de informação já documentada na cascata de
`lambda_estimator.py` (Nível A melhor que B, melhor que C/D) — não um
rótulo adicional.
"""

import unittest

from src.engine.decision import DecisionEngine
from src.engine.dixon_coles import calculate_fractional_kelly
from src.engine.kelly import (
    calculate_adaptive_kelly_fraction,
    calculate_confidence_multiplier,
    fractional_kelly,
    kelly_fraction,
)
from src.engine.lambda_estimator import SHRINKAGE_K
from src.backtest.historical.evaluator import evaluate_bet
from src.backtest.historical.models import HistoricalBet
from src.backtest.historical.staking import FlatStake, KellyStake


class TestConfidenceMultiplierMissingMetadata(unittest.TestCase):
    """Ausência dos metadados de confiança -> multiplicador == 1.0 (sem escala)."""

    def test_both_none_returns_one(self):
        self.assertEqual(calculate_confidence_multiplier(None, None), 1.0)

    def test_only_tier_none_returns_one(self):
        self.assertEqual(calculate_confidence_multiplier(None, 10.0), 1.0)

    def test_only_sample_size_none_returns_one(self):
        self.assertEqual(calculate_confidence_multiplier("recent_matches", None), 1.0)

    def test_nan_sample_size_returns_one(self):
        self.assertEqual(calculate_confidence_multiplier("recent_matches", float("nan")), 1.0)

    def test_non_numeric_sample_size_returns_one(self):
        self.assertEqual(calculate_confidence_multiplier("recent_matches", "invalid"), 1.0)

    def test_adaptive_fraction_equals_base_fraction_without_metadata(self):
        self.assertEqual(calculate_adaptive_kelly_fraction(0.25), 0.25)
        self.assertEqual(calculate_adaptive_kelly_fraction(0.25, None, None), 0.25)


class TestConfidenceMultiplierSampleSize(unittest.TestCase):
    """Comportamento do multiplicador conforme `effective_sample_size`."""

    def test_very_small_effective_sample_size_gives_small_multiplier(self):
        multiplier = calculate_confidence_multiplier("avg_total_goals_or_prior", 1.0)
        self.assertGreater(multiplier, 0.0)
        self.assertLess(multiplier, 0.15)

    def test_zero_effective_sample_size_is_lower_bound(self):
        multiplier = calculate_confidence_multiplier("recent_matches", 0.0)
        self.assertEqual(multiplier, 0.0)

    def test_negative_effective_sample_size_is_treated_as_zero(self):
        multiplier = calculate_confidence_multiplier("recent_matches", -5.0)
        self.assertEqual(multiplier, 0.0)

    def test_large_effective_sample_size_approaches_but_never_reaches_one(self):
        multiplier = calculate_confidence_multiplier("recent_matches", 1_000_000.0)
        self.assertLess(multiplier, 1.0)
        self.assertGreater(multiplier, 0.999)

    def test_multiplier_is_monotonically_increasing_with_sample_size(self):
        samples = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]
        multipliers = [
            calculate_confidence_multiplier("recent_matches", n) for n in samples
        ]
        for earlier, later in zip(multipliers, multipliers[1:]):
            self.assertLess(earlier, later)

    def test_multiplier_has_no_jump_across_the_full_sample_size_range(self):
        # A função tem de ser contínua em `effective_sample_size` -- sem
        # "saltos" para nenhum valor de tier, em toda a gama observável.
        # (Isto teria falhado com um desenho que derivasse a escala de
        # confiança a partir de `classify_model_confidence`, cujas
        # fronteiras dependem do próprio `effective_sample_size`.)
        for tier in ("recent_matches", "h2h_goal_totals", "avg_total_goals_or_prior"):
            step = 0.01
            previous = calculate_confidence_multiplier(tier, 0.0)
            n_eff = 0.0
            for _ in range(3000):
                n_eff += step
                current = calculate_confidence_multiplier(tier, n_eff)
                self.assertLess(abs(current - previous), 0.01, msg=f"jump at tier={tier}, n_eff={n_eff}")
                previous = current


class TestConfidenceMultiplierByTier(unittest.TestCase):
    """Tier HIGH ("recent_matches") / MEDIUM ("h2h_goal_totals") / LOW ("avg_total_goals_or_prior")."""

    def test_tier_high_recent_matches_formula(self):
        multiplier = calculate_confidence_multiplier("recent_matches", 9.0)
        self.assertAlmostEqual(multiplier, 9.0 / (9.0 + SHRINKAGE_K), places=9)

    def test_tier_medium_h2h_goal_totals_formula(self):
        multiplier = calculate_confidence_multiplier("h2h_goal_totals", 9.0)
        self.assertAlmostEqual(multiplier, 9.0 / (9.0 + 2 * SHRINKAGE_K), places=9)

    def test_tier_low_avg_total_goals_or_prior_formula(self):
        multiplier = calculate_confidence_multiplier("avg_total_goals_or_prior", 9.0)
        self.assertAlmostEqual(multiplier, 9.0 / (9.0 + 4 * SHRINKAGE_K), places=9)

    def test_tier_ordering_at_the_same_sample_size(self):
        # Mesma amostra efetiva, tier melhor -> multiplicador maior.
        n_eff = 9.0
        high = calculate_confidence_multiplier("recent_matches", n_eff)
        medium = calculate_confidence_multiplier("h2h_goal_totals", n_eff)
        low = calculate_confidence_multiplier("avg_total_goals_or_prior", n_eff)
        self.assertGreater(high, medium)
        self.assertGreater(medium, low)

    def test_unknown_tier_falls_back_to_the_most_conservative_known_tier(self):
        unknown = calculate_confidence_multiplier("some_future_tier", 9.0)
        low = calculate_confidence_multiplier("avg_total_goals_or_prior", 9.0)
        self.assertEqual(unknown, low)


class TestAdaptiveFractionBounds(unittest.TestCase):
    """Continuidade e limites da fração adaptativa (item 3/4 da melhoria)."""

    def test_adaptive_fraction_never_exceeds_base_fraction(self):
        for tier in ("recent_matches", "h2h_goal_totals", "avg_total_goals_or_prior"):
            for n_eff in (0.0, 0.5, 1, 2, 4, 8, 16, 100, 10_000):
                fraction = calculate_adaptive_kelly_fraction(0.25, tier, n_eff)
                self.assertLessEqual(fraction, 0.25)
                self.assertGreaterEqual(fraction, 0.0)

    def test_adaptive_fraction_is_continuous_function_of_sample_size(self):
        step = 0.001
        previous = calculate_adaptive_kelly_fraction(0.25, "recent_matches", 5.0)
        for i in range(1, 2000):
            n_eff = 5.0 + i * step
            current = calculate_adaptive_kelly_fraction(0.25, "recent_matches", n_eff)
            self.assertLess(abs(current - previous), 0.001)
            previous = current


class TestFractionalKellyBackwardCompatibility(unittest.TestCase):
    """`fractional_kelly` sem metadados == comportamento anterior à Melhoria #6."""

    def test_matches_kelly_fraction_times_default_fraction(self):
        p, o = 0.55, 2.10
        self.assertAlmostEqual(
            fractional_kelly(p, o), kelly_fraction(p, o) * 0.25, places=9
        )

    def test_matches_explicit_fraction_without_metadata(self):
        p, o = 0.6, 3.0
        self.assertAlmostEqual(
            fractional_kelly(p, o, fraction=0.5),
            kelly_fraction(p, o) * 0.5,
            places=9,
        )

    def test_known_reference_value_unchanged(self):
        # Valor de referência documentado em src/tools/test_kelly.py, já
        # existente antes desta melhoria.
        stake = fractional_kelly(0.55, 2.10)
        self.assertAlmostEqual(round(stake * 100, 2), 3.52, places=2)

    def test_with_metadata_stake_never_exceeds_stake_without_metadata(self):
        p, o = 0.55, 2.10
        baseline = fractional_kelly(p, o)
        scaled = fractional_kelly(
            p, o, lambda_tier="avg_total_goals_or_prior", effective_sample_size=1.0
        )
        self.assertLess(scaled, baseline)


class TestDixonColesRegression(unittest.TestCase):
    """`calculate_fractional_kelly` (dixon_coles.py) — unificado, sem duplicação."""

    def test_without_metadata_matches_kelly_module_formula(self):
        prob, odds = 0.55, 2.10
        expected = round(min(kelly_fraction(prob, odds) * 0.25, 0.02), 4)
        self.assertAlmostEqual(calculate_fractional_kelly(prob, odds), expected, places=6)

    def test_guard_clauses_preserved(self):
        self.assertEqual(calculate_fractional_kelly(0.5, 1.0), 0.0)
        self.assertEqual(calculate_fractional_kelly(0.0, 2.0), 0.0)
        self.assertEqual(calculate_fractional_kelly(0.1, 1.5), 0.0)  # kelly negativo

    def test_cap_still_enforced_with_confidence_metadata(self):
        stake = calculate_fractional_kelly(
            0.95, 5.0, lambda_tier="recent_matches", effective_sample_size=1_000_000.0
        )
        self.assertLessEqual(stake, 0.02)

    def test_low_confidence_reduces_stake_below_uncapped_baseline(self):
        prob, odds = 0.6, 2.5
        baseline = calculate_fractional_kelly(prob, odds, max_stake_pct=1.0)
        scaled = calculate_fractional_kelly(
            prob, odds, max_stake_pct=1.0,
            lambda_tier="avg_total_goals_or_prior", effective_sample_size=1.0,
        )
        self.assertLess(scaled, baseline)


class TestDecisionEngineRegression(unittest.TestCase):
    """`DecisionEngine.evaluate_bet` — unificado, sem recalcular Kelly localmente."""

    def test_without_metadata_matches_previous_formula(self):
        engine = DecisionEngine(max_kelly_fraction=0.25, min_edge=5.0)
        rec = engine.evaluate_bet("HOME", 60.0, 2.5)

        b = 2.5 - 1.0
        p = 0.6
        q = 1.0 - p
        expected_full_kelly = max((b * p - q) / b, 0)
        expected_stake = min(expected_full_kelly * 0.25 * 100.0, 5.0)

        self.assertAlmostEqual(rec.kelly_stake_pct, round(expected_stake, 2), places=6)
        self.assertEqual(rec.action, "BET 🔥")

    def test_pass_below_min_edge_unaffected_by_metadata(self):
        engine = DecisionEngine(max_kelly_fraction=0.25, min_edge=50.0)
        rec = engine.evaluate_bet(
            "HOME", 51.0, 1.90,
            lambda_tier="recent_matches", effective_sample_size=100.0,
        )
        self.assertEqual(rec.action, "PASS ❄️")
        self.assertEqual(rec.kelly_stake_pct, 0.0)

    def test_low_confidence_reduces_suggested_stake(self):
        engine = DecisionEngine(max_kelly_fraction=0.25, min_edge=5.0)
        baseline = engine.evaluate_bet("HOME", 60.0, 2.5)
        low_confidence = engine.evaluate_bet(
            "HOME", 60.0, 2.5,
            lambda_tier="avg_total_goals_or_prior", effective_sample_size=1.0,
        )
        self.assertLess(low_confidence.kelly_stake_pct, baseline.kelly_stake_pct)
        self.assertGreater(low_confidence.kelly_stake_pct, 0.0)

    def test_high_confidence_stays_within_original_hard_cap(self):
        engine = DecisionEngine(max_kelly_fraction=0.25, min_edge=5.0)
        rec = engine.evaluate_bet(
            "HOME", 95.0, 5.0,
            lambda_tier="recent_matches", effective_sample_size=1_000_000.0,
        )
        self.assertLessEqual(rec.kelly_stake_pct, 5.0)  # cap de 5% preservado


class TestKellyStakeBacktestIntegration(unittest.TestCase):
    """`KellyStake` / `evaluate_bet` do Backtesting Framework."""

    def _bet(self, **kwargs):
        return HistoricalBet(
            match="A vs B",
            date="2026-01-01",
            market="HOME",
            odd=2.5,
            model_prob=0.6,
            engine_decision="BET",
            result="WIN",
            **kwargs,
        )

    def test_stake_for_without_metadata_matches_previous_behavior(self):
        staking = KellyStake(fraction=0.25, cap=0.05, bankroll=100.0)
        stake_new_signature = staking.stake_for(0.6, 2.5)
        expected = round(min(fractional_kelly(0.6, 2.5), 0.05) * 100.0, 4)
        self.assertAlmostEqual(stake_new_signature, expected, places=6)

    def test_flat_stake_ignores_confidence_metadata(self):
        staking = FlatStake(unit=2.0)
        stake = staking.stake_for(
            0.6, 2.5, lambda_tier="avg_total_goals_or_prior", effective_sample_size=1.0
        )
        self.assertEqual(stake, 2.0)

    def test_evaluate_bet_without_confidence_metadata_is_unchanged(self):
        bet = self._bet()
        self.assertIsNone(bet.lambda_tier)
        self.assertIsNone(bet.effective_sample_size)
        evaluated = evaluate_bet(bet, staking=KellyStake(fraction=0.25, cap=0.05, bankroll=100.0))
        expected = round(min(fractional_kelly(0.6, 2.5), 0.05) * 100.0, 4)
        self.assertAlmostEqual(evaluated.stake, expected, places=6)

    def test_evaluate_bet_low_confidence_reduces_hypothetical_stake(self):
        confident_bet = self._bet(lambda_tier="recent_matches", effective_sample_size=1_000_000.0)
        unconfident_bet = self._bet(
            lambda_tier="avg_total_goals_or_prior", effective_sample_size=1.0
        )
        staking = KellyStake(fraction=0.25, cap=0.05, bankroll=100.0)
        confident_eval = evaluate_bet(confident_bet, staking=staking)
        unconfident_eval = evaluate_bet(unconfident_bet, staking=staking)

        self.assertGreater(unconfident_eval.stake, 0.0)
        self.assertLess(unconfident_eval.stake, confident_eval.stake)
        # `kelly` (Kelly completo, sem fração) não é afetado pela confiança —
        # só `stake` muda.
        self.assertAlmostEqual(confident_eval.kelly, unconfident_eval.kelly, places=9)

    def test_engine_decision_and_probability_unaffected_by_confidence(self):
        # Critério de seleção de apostas (engine_decision/placed) e
        # probabilidade/edge/ev não podem mudar com a confiança -- só stake.
        confident_bet = self._bet(lambda_tier="recent_matches", effective_sample_size=1_000_000.0)
        unconfident_bet = self._bet(
            lambda_tier="avg_total_goals_or_prior", effective_sample_size=1.0
        )
        staking = KellyStake(fraction=0.25, cap=0.05, bankroll=100.0)
        confident_eval = evaluate_bet(confident_bet, staking=staking)
        unconfident_eval = evaluate_bet(unconfident_bet, staking=staking)

        self.assertEqual(confident_eval.placed, unconfident_eval.placed)
        self.assertEqual(confident_eval.engine_decision, unconfident_eval.engine_decision)
        self.assertAlmostEqual(confident_eval.probability, unconfident_eval.probability, places=9)
        self.assertAlmostEqual(confident_eval.edge, unconfident_eval.edge, places=9)
        self.assertAlmostEqual(confident_eval.ev, unconfident_eval.ev, places=9)


class TestCrossImplementationConsistency(unittest.TestCase):
    """Todas as implementações de Kelly usam agora a mesma fração adaptativa."""

    def test_kelly_dixon_coles_and_decision_agree_without_metadata(self):
        # `prob`/`odd` escolhidos para que o Kelly fracionário (1/4) fique
        # abaixo do cap de 5% hard-coded do DecisionEngine (§7.3 da
        # auditoria) — esse cap é uma proteção adicional e distinta da
        # fração adaptativa, preservada tal e qual por esta melhoria.
        prob, odd = 0.55, 2.10

        via_kelly = fractional_kelly(prob, odd, fraction=0.25)
        via_dixon_coles = calculate_fractional_kelly(prob, odd, fraction=0.25, max_stake_pct=1.0)

        engine = DecisionEngine(max_kelly_fraction=0.25, min_edge=0.0)
        rec = engine.evaluate_bet("HOME", prob * 100.0, odd)
        via_decision = rec.kelly_stake_pct / 100.0

        self.assertAlmostEqual(via_kelly, via_dixon_coles, places=4)
        self.assertAlmostEqual(via_kelly, via_decision, places=4)

    def test_kelly_and_dixon_coles_agree_with_same_confidence_metadata(self):
        prob, odd = 0.6, 2.5
        via_kelly = fractional_kelly(
            prob, odd, fraction=0.25,
            lambda_tier="recent_matches", effective_sample_size=5.0,
        )
        via_dixon_coles = calculate_fractional_kelly(
            prob, odd, fraction=0.25, max_stake_pct=1.0,
            lambda_tier="recent_matches", effective_sample_size=5.0,
        )
        self.assertAlmostEqual(via_kelly, via_dixon_coles, places=4)


if __name__ == "__main__":
    unittest.main()
