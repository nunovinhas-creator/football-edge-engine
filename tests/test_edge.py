"""
Testes unitários da definição oficial de Edge (src/engine/edge.py) e dos
módulos que a consomem.

Cobrem:
  - cálculo correto de Edge e EV (valores conhecidos);
  - casos limite (probabilidade == 1.0, odd próxima de 1.0);
  - odds inválidas (<= 1.0, zero, negativas);
  - probabilidades inválidas (<= 0.0, > 1.0, escala percentual por engano);
  - comportamento pós-correção do bug em src/engine/market.py e no
    caminho ativo `main.py predict` (src/cli/predict.py);
  - consistência entre todos os módulos que reutilizam calculate_edge.
"""

import unittest

from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev,
    edge,
    expected_value,
)
from src.engine.market import analyze_market
from src.engine.analyzer import analyze_bet
from src.engine.decision import DecisionEngine
from src.engine.live_decision import evaluate_live_market
from src.engine.bet_engine import evaluate_bet as bet_engine_evaluate_bet


class TestImpliedProbability(unittest.TestCase):

    def test_normal_odd(self):
        self.assertEqual(implied_probability(2.10), 0.4762)

    def test_odd_equal_to_one_is_invalid_returns_zero(self):
        self.assertEqual(implied_probability(1.0), 0.0)

    def test_odd_below_one_is_invalid_returns_zero(self):
        self.assertEqual(implied_probability(0.5), 0.0)


class TestCalculateEdgeOfficial(unittest.TestCase):
    """
    Valida a fórmula oficial: edge = prob_model - implied_probability(odd_house).
    Os valores usados (p=0.55, odd=2.10 -> edge=0.0738 / ev=0.155) já eram os
    fixtures usados nos scripts de demonstração do repositório
    (src/tools/test_report.py, test_ranking.py, test_stake.py, test_filter.py:
    edge=7.38%, ev=15.5%), confirmando que esta é a fórmula com que o resto
    do sistema já era coerente.
    """

    def test_known_value(self):
        result = calculate_edge(0.55, 2.10)
        self.assertAlmostEqual(result, 0.0738, places=4)
        self.assertAlmostEqual(round(result * 100, 2), 7.38)

    def test_edge_and_ev_are_different_quantities(self):
        p, odd = 0.55, 2.10
        self.assertNotAlmostEqual(calculate_edge(p, odd), calculate_ev(p, odd))
        self.assertAlmostEqual(round(calculate_ev(p, odd) * 100, 2), 15.5)

    def test_no_edge_when_model_matches_market(self):
        odd = 2.0
        fair_prob = implied_probability(odd)
        self.assertAlmostEqual(calculate_edge(fair_prob, odd), 0.0, places=4)

    def test_wrapper_alias_matches_calculate_edge(self):
        self.assertEqual(edge(0.55, 2.10), calculate_edge(0.55, 2.10))

    def test_expected_value_alias_matches_calculate_ev_not_edge(self):
        # Regressão: antes desta correção, `expected_value` estava (por bug)
        # ligado à mesma fórmula de `calculate_edge`, tornando-o idêntico ao
        # edge em vez de ao EV.
        self.assertEqual(expected_value(0.55, 2.10), calculate_ev(0.55, 2.10))
        self.assertNotEqual(expected_value(0.55, 2.10), edge(0.55, 2.10))

    # --- Casos limite ---

    def test_prob_model_boundary_one_is_valid(self):
        result = calculate_edge(1.0, 2.10)
        self.assertAlmostEqual(result, 1.0 - implied_probability(2.10), places=4)

    def test_odd_just_above_one_is_valid(self):
        result = calculate_edge(0.5, 1.0001)
        self.assertIsInstance(result, float)

    # --- Odds inválidas ---

    def test_odd_equal_to_one_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.5, 1.0)

    def test_odd_below_one_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.5, 0.8)

    def test_odd_zero_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.5, 0.0)

    def test_odd_negative_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.5, -2.5)

    # --- Probabilidades inválidas ---

    def test_prob_zero_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.0, 2.10)

    def test_prob_negative_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(-0.1, 2.10)

    def test_prob_above_one_raises(self):
        with self.assertRaises(ValueError):
            calculate_edge(1.5, 2.10)

    def test_prob_passed_as_raw_percentage_raises(self):
        # Este é exatamente o formato do bug histórico: alguém passa a
        # probabilidade em escala 0-100 (ex. 55) em vez de fração (0.55).
        with self.assertRaises(ValueError):
            calculate_edge(55, 2.10)

    def test_the_historical_bug_shape_now_raises_instead_of_silently_wrong(self):
        # Reprodução exata do bug de market.py: chamar calculate_edge com uma
        # probabilidade de mercado (0.0-1.0) no lugar da odd.
        market_probability = implied_probability(2.10)  # 0.4762
        with self.assertRaises(ValueError):
            calculate_edge(0.55, market_probability)


class TestCalculateEvUnchanged(unittest.TestCase):
    """calculate_ev mantém o comportamento e a fórmula documentados no
    audit como corretos — não foi alterado por esta correção."""

    def test_known_value(self):
        self.assertAlmostEqual(calculate_ev(0.55, 2.10), 0.155, places=4)

    def test_invalid_odd_returns_sentinel_not_exception(self):
        self.assertEqual(calculate_ev(0.5, 1.0), -1.0)

    def test_invalid_prob_returns_sentinel_not_exception(self):
        self.assertEqual(calculate_ev(0.0, 2.10), -1.0)


class TestMarketBugFix(unittest.TestCase):
    """Regressão do bug documentado em docs/AUDIT_MATEMATICA.md §6.3:
    analyze_market() chamava calculate_edge(model_probability, market_probability)
    em vez de calculate_edge(model_probability, odd), devolvendo -1.0 sempre."""

    def setUp(self):
        self.odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        self.model_probability = 0.55  # fração, conforme contrato oficial

    def test_edge_is_not_always_minus_one(self):
        result = analyze_market(self.odds, self.model_probability)
        edges = [outcome["edge"] for outcome in result.values()]
        self.assertFalse(all(e == -1.0 for e in edges))

    def test_edge_matches_official_formula(self):
        result = analyze_market(self.odds, self.model_probability)
        for outcome_name, odd in self.odds.items():
            expected = calculate_edge(self.model_probability, odd)
            self.assertEqual(result[outcome_name]["edge"], expected)

    def test_ev_untouched_and_correct(self):
        result = analyze_market(self.odds, self.model_probability)
        for outcome_name, odd in self.odds.items():
            expected = calculate_ev(self.model_probability, odd)
            self.assertEqual(result[outcome_name]["ev"], expected)


class TestAnalyzerBetScaleFix(unittest.TestCase):
    """analyze_bet() recebe model_probability em escala percentual (0-100),
    tal como devolvido por predict_probability()/Match.probability. Antes
    da correção, essa probabilidade não era normalizada antes de entrar em
    calculate_edge/calculate_ev, produzindo edge/ev sistematicamente
    inflacionados (e, após esta correção do módulo edge.py, um ValueError)."""

    def test_matches_reference_fixture(self):
        # p=55%, odd=2.10 -> edge=7.38%, ev=15.5% (fixtures usados em
        # src/tools/test_report.py, test_ranking.py, test_filter.py, test_stake.py)
        result = analyze_bet(2.10, 55)
        self.assertAlmostEqual(result["edge"], 7.38, places=2)
        self.assertAlmostEqual(result["ev"], 15.5, places=1)

    def test_invalid_probability_does_not_crash(self):
        result = analyze_bet(2.10, 0)
        self.assertEqual(result["decision"], "PASS")

    def test_invalid_odd_does_not_crash(self):
        result = analyze_bet(1.0, 55)
        self.assertEqual(result["decision"], "PASS")


class TestDecisionEngineUsesOfficialEdge(unittest.TestCase):

    def test_edge_pct_matches_central_function(self):
        engine = DecisionEngine()
        rec = engine.evaluate_bet("HOME", 55, 2.10)
        expected_edge_pct = round(calculate_edge(0.55, 2.10) * 100, 1)
        self.assertEqual(rec.edge_pct, expected_edge_pct)

    def test_invalid_probability_above_100_pct_is_safe(self):
        engine = DecisionEngine()
        rec = engine.evaluate_bet("HOME", 150, 2.10)
        self.assertEqual(rec.action, "PASS")

    def test_odd_at_or_below_one_is_safe(self):
        engine = DecisionEngine()
        rec = engine.evaluate_bet("HOME", 55, 1.0)
        self.assertEqual(rec.action, "PASS")


class TestLiveDecisionUsesOfficialEdge(unittest.TestCase):

    def test_edge_matches_central_function(self):
        result = evaluate_live_market(probability_pct=72, bookie_odd=2.10, market="NEXT GOAL")
        expected_edge = round(calculate_edge(0.72, 2.10) * 100, 2)
        self.assertEqual(result.edge, expected_edge)

    def test_matches_previous_behaviour_for_valid_inputs(self):
        # A fórmula anterior (probability_pct - (1/odd)*100) e a nova
        # (calculate_edge(p, odd)*100) devem coincidir para odds/probabilidades
        # normais — esta é uma refatoração de centralização, não uma mudança
        # de comportamento.
        probability_pct, bookie_odd = 72, 2.10
        old_style_edge = probability_pct - (1 / bookie_odd) * 100
        result = evaluate_live_market(probability_pct=probability_pct, bookie_odd=bookie_odd)
        self.assertAlmostEqual(result.edge, old_style_edge, places=1)

    def test_invalid_odd_does_not_raise(self):
        result = evaluate_live_market(probability_pct=72, bookie_odd=1.0)
        self.assertEqual(result.action, "❄️ PASS")

    def test_invalid_odd_zero_does_not_raise_zero_division(self):
        result = evaluate_live_market(probability_pct=72, bookie_odd=0.0)
        self.assertEqual(result.action, "❄️ PASS")


class TestBetEngineUsesOfficialEdge(unittest.TestCase):

    def test_matches_central_function(self):
        result = bet_engine_evaluate_bet(odd=2.10, model_probability=55)
        expected_edge = round(calculate_edge(0.55, 2.10) * 100, 2)
        expected_ev = round(calculate_ev(0.55, 2.10) * 100, 2)
        self.assertEqual(result["edge"], expected_edge)
        self.assertEqual(result["ev"], expected_ev)

    def test_invalid_odd_is_safe(self):
        result = bet_engine_evaluate_bet(odd=1.0, model_probability=55)
        self.assertEqual(result["value"], False)
        self.assertEqual(result["edge"], -100.0)


if __name__ == "__main__":
    unittest.main()
