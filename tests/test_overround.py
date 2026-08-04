"""
Testes da remoção do overround (margem da casa) antes do cálculo de Edge e
Expected Value (EV) — Melhoria #7 da auditoria matemática
(docs/AUDIT_MATEMATICA.md).

Cobrem:
  - `remove_overround()` (a função única e reutilizável) para os mercados
    1X2, Over/Under e BTTS;
  - overround elevado vs. overround baixo;
  - ausência de odds suficientes (fallback para o comportamento atual);
  - `calculate_edge()`/`calculate_ev()` com e sem `market_odds`;
  - retrocompatibilidade: chamadas antigas (sem `market_odds`) continuam a
    devolver exatamente os mesmos valores de antes desta melhoria;
  - regressão: módulos que não foram alterados (Kelly, DecisionEngine,
    live_decision, bet_engine, analyzer) continuam a produzir os mesmos
    números de sempre, porque não passam `market_odds`.
"""

import unittest

from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev,
    remove_overround,
)
from src.engine.market import analyze_market
from src.engine.analyzer import analyze_bet
from src.engine.decision import DecisionEngine
from src.engine.live_decision import evaluate_live_market
from src.engine.bet_engine import evaluate_bet as bet_engine_evaluate_bet
from src.engine.kelly import fractional_kelly


class TestRemoveOverroundBasics(unittest.TestCase):

    def test_returns_none_with_zero_odds(self):
        self.assertIsNone(remove_overround([]))

    def test_returns_none_with_only_one_valid_odd(self):
        self.assertIsNone(remove_overround([2.10]))
        self.assertIsNone(remove_overround({"HOME": 2.10}))

    def test_returns_none_with_one_valid_and_one_invalid_odd(self):
        # Apenas uma odd realmente utilizável (a outra é <= 1.0) — não há
        # mercado suficiente para calcular overround com significado.
        self.assertIsNone(remove_overround([2.10, 1.0]))
        self.assertIsNone(remove_overround({"HOME": 2.10, "AWAY": 0.0}))

    def test_fair_probabilities_sum_to_one(self):
        fair = remove_overround({"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60})
        self.assertIsNotNone(fair)
        self.assertAlmostEqual(sum(fair.values()), 1.0, places=6)

    def test_preserves_dict_shape_and_keys(self):
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        fair = remove_overround(odds)
        self.assertEqual(set(fair.keys()), set(odds.keys()))

    def test_preserves_list_shape_and_length(self):
        odds = [1.65, 4.20, 4.60]
        fair = remove_overround(odds)
        self.assertEqual(len(fair), len(odds))

    def test_fair_probability_lower_than_implied_when_margin_present(self):
        # Overround > 1.0 (margem positiva) -> a probabilidade fair de cada
        # opção é sempre menor que a implícita (com margem).
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        fair = remove_overround(odds)
        for outcome, odd in odds.items():
            self.assertLess(fair[outcome], implied_probability(odd))


class TestRemoveOverround1X2Market(unittest.TestCase):
    """Mercado 1X2 com margem realista (~6%)."""

    def setUp(self):
        self.odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        implied = {k: 1.0 / v for k, v in self.odds.items()}
        self.overround = sum(implied.values())
        self.expected_fair = {k: v / self.overround for k, v in implied.items()}

    def test_matches_hand_computed_values(self):
        fair = remove_overround(self.odds)
        for outcome in self.odds:
            self.assertAlmostEqual(fair[outcome], self.expected_fair[outcome], places=6)

    def test_overround_is_above_one(self):
        self.assertGreater(self.overround, 1.0)


class TestRemoveOverroundOverUnderMarket(unittest.TestCase):
    """Mercado Over/Under 2.5 golos, odds simétricas (~5.26% de margem)."""

    def test_symmetric_odds_produce_equal_fair_probabilities(self):
        odds = {"OVER_2.5": 1.90, "UNDER_2.5": 1.90}
        fair = remove_overround(odds)
        self.assertAlmostEqual(fair["OVER_2.5"], 0.5, places=6)
        self.assertAlmostEqual(fair["UNDER_2.5"], 0.5, places=6)
        self.assertAlmostEqual(sum(fair.values()), 1.0, places=9)


class TestRemoveOverroundBttsMarket(unittest.TestCase):
    """Mercado BTTS (Ambas Marcam) Sim/Não."""

    def test_matches_hand_computed_values(self):
        odds = {"YES": 1.80, "NO": 2.00}
        implied_yes = 1.0 / 1.80
        implied_no = 1.0 / 2.00
        overround = implied_yes + implied_no
        fair = remove_overround(odds)
        self.assertAlmostEqual(fair["YES"], implied_yes / overround, places=6)
        self.assertAlmostEqual(fair["NO"], implied_no / overround, places=6)


class TestOverroundMagnitude(unittest.TestCase):
    """Overround elevado vs. overround baixo, sobre o mesmo par de
    probabilidades fair subjacentes (0.5/0.5), para isolar o efeito da
    margem em si."""

    def test_high_overround_market(self):
        # implied = 0.4 / 0.4 / 0.4 -> overround = 1.2 (20% de margem)
        odds = {"A": 2.5, "B": 2.5, "C": 2.5}
        fair = remove_overround(odds)
        for outcome in odds:
            self.assertAlmostEqual(fair[outcome], 1.0 / 3.0, places=6)

    def test_low_overround_market(self):
        # implied = 0.51 / 0.51 -> overround = 1.02 (2% de margem)
        odds = {"HOME": 1.0 / 0.51, "AWAY": 1.0 / 0.51}
        fair = remove_overround(odds)
        for outcome in odds:
            self.assertAlmostEqual(fair[outcome], 0.5, places=6)

    def test_fair_edge_is_invariant_to_margin_magnitude(self):
        # Mesma probabilidade "fair" subjacente (0.5) em ambos os mercados,
        # apenas com margens diferentes (10% vs 2%). Com o overround
        # removido, o edge resultante deve ser o MESMO nos dois casos —
        # a margem do bookmaker deixa de influenciar o edge calculado.
        high_margin_odd = 1.0 / (0.5 * 1.10)  # ~10% de overround -> odd ~1.818
        low_margin_odd = 1.0 / (0.5 * 1.02)   # ~2% de overround -> odd ~1.961

        high_margin_odds = {"A": high_margin_odd, "B": high_margin_odd}
        low_margin_odds = {"HOME": low_margin_odd, "AWAY": low_margin_odd}

        prob_model = 0.60

        edge_high = calculate_edge(prob_model, high_margin_odd, market_odds=high_margin_odds)
        edge_low = calculate_edge(prob_model, low_margin_odd, market_odds=low_margin_odds)

        self.assertAlmostEqual(edge_high, edge_low, places=4)
        self.assertAlmostEqual(edge_high, round(prob_model - 0.5, 4), places=4)

    def test_without_removal_higher_margin_produces_lower_apparent_edge(self):
        # Sem remoção de overround (comportamento atual, sem market_odds),
        # a mesma probabilidade fair subjacente (0.5) mas com margem maior
        # produz um edge aparente MENOR — é exatamente esta distorção que a
        # Melhoria #7 corrige quando market_odds está disponível.
        high_margin_odd = 1.0 / (0.5 * 1.10)
        low_margin_odd = 1.0 / (0.5 * 1.02)
        prob_model = 0.60

        old_edge_high_margin = calculate_edge(prob_model, high_margin_odd)
        old_edge_low_margin = calculate_edge(prob_model, low_margin_odd)

        self.assertLess(old_edge_high_margin, old_edge_low_margin)


class TestCalculateEdgeWithMarketOdds(unittest.TestCase):

    def test_uses_fair_probability_when_market_odds_given(self):
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        fair = remove_overround(odds)

        result = calculate_edge(0.55, 1.65, market_odds=odds)
        expected = round(0.55 - fair["HOME"], 4)
        self.assertEqual(result, expected)

    def test_falls_back_to_implied_probability_when_market_odds_none(self):
        result = calculate_edge(0.55, 2.10, market_odds=None)
        expected = round(0.55 - implied_probability(2.10), 4)
        self.assertEqual(result, expected)

    def test_falls_back_when_market_odds_insufficient(self):
        # Só uma odd válida no "mercado" -> não há overround calculável.
        result = calculate_edge(0.55, 2.10, market_odds={"HOME": 2.10})
        expected = calculate_edge(0.55, 2.10)
        self.assertEqual(result, expected)

    def test_falls_back_when_market_odds_empty(self):
        result = calculate_edge(0.55, 2.10, market_odds={})
        expected = calculate_edge(0.55, 2.10)
        self.assertEqual(result, expected)

    def test_falls_back_when_odd_house_not_in_market_odds(self):
        # odd_house (2.10) nao pertence ao conjunto de market_odds fornecido.
        result = calculate_edge(0.55, 2.10, market_odds={"HOME": 1.65, "AWAY": 4.60})
        expected = calculate_edge(0.55, 2.10)
        self.assertEqual(result, expected)

    def test_accepts_list_of_odds_not_only_dict(self):
        odds_list = [1.65, 4.20, 4.60]
        result_dict = calculate_edge(0.55, 1.65, market_odds={"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60})
        result_list = calculate_edge(0.55, 1.65, market_odds=odds_list)
        self.assertEqual(result_dict, result_list)

    def test_still_raises_for_invalid_prob_model_regardless_of_market_odds(self):
        with self.assertRaises(ValueError):
            calculate_edge(1.5, 2.10, market_odds={"HOME": 2.10, "AWAY": 1.80})

    def test_still_raises_for_invalid_odd_house_regardless_of_market_odds(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.55, 0.8, market_odds={"HOME": 0.8, "AWAY": 1.80})


class TestCalculateEvUnaffectedByOverroundRemoval(unittest.TestCase):
    """Confirma o achado documentado em docs/AUDIT_MATEMATICA.md §6.1/§11:
    a fórmula de EV (p*odd - 1) não depende da probabilidade implícita de
    mercado, pelo que remover o overround não pode alterar o seu valor."""

    def test_ev_identical_with_and_without_market_odds(self):
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        ev_without = calculate_ev(0.55, 1.65)
        ev_with = calculate_ev(0.55, 1.65, market_odds=odds)
        self.assertEqual(ev_without, ev_with)

    def test_ev_matches_known_fixture_regardless_of_market_odds(self):
        self.assertAlmostEqual(
            calculate_ev(0.55, 2.10, market_odds={"HOME": 2.10, "AWAY": 1.80}),
            0.155,
            places=4,
        )


class TestBackwardCompatibility(unittest.TestCase):
    """Chamadas no formato antigo (2 argumentos posicionais, sem
    market_odds) devem continuar a devolver exatamente os mesmos valores
    de antes desta melhoria."""

    def test_calculate_edge_two_args_matches_pre_melhoria_fixture(self):
        result = calculate_edge(0.55, 2.10)
        self.assertAlmostEqual(result, 0.0738, places=4)

    def test_calculate_ev_two_args_matches_pre_melhoria_fixture(self):
        result = calculate_ev(0.55, 2.10)
        self.assertAlmostEqual(result, 0.155, places=4)

    def test_calculate_edge_still_raises_on_invalid_input(self):
        with self.assertRaises(ValueError):
            calculate_edge(0.5, 1.0)

    def test_calculate_ev_still_returns_sentinel_on_invalid_input(self):
        self.assertEqual(calculate_ev(0.5, 1.0), -1.0)

    def test_implied_probability_unchanged(self):
        self.assertEqual(implied_probability(2.10), 0.4762)


class TestRegressionOfUnchangedModules(unittest.TestCase):
    """Módulos que esta melhoria não altera (Kelly, DecisionEngine,
    live_decision, bet_engine, analyzer) — porque não passam `market_odds`
    — continuam a produzir exatamente os mesmos números de sempre."""

    def test_kelly_unaffected(self):
        self.assertAlmostEqual(fractional_kelly(0.55, 2.10), fractional_kelly(0.55, 2.10))

    def test_decision_engine_unaffected(self):
        engine = DecisionEngine()
        rec = engine.evaluate_bet("HOME", 55, 2.10)
        expected_edge_pct = round(calculate_edge(0.55, 2.10) * 100, 1)
        self.assertEqual(rec.edge_pct, expected_edge_pct)

    def test_live_decision_unaffected(self):
        result = evaluate_live_market(probability_pct=72, bookie_odd=2.10, market="NEXT GOAL")
        expected_edge = round(calculate_edge(0.72, 2.10) * 100, 2)
        self.assertEqual(result.edge, expected_edge)

    def test_bet_engine_unaffected(self):
        result = bet_engine_evaluate_bet(odd=2.10, model_probability=55)
        expected_edge = round(calculate_edge(0.55, 2.10) * 100, 2)
        expected_ev = round(calculate_ev(0.55, 2.10) * 100, 2)
        self.assertEqual(result["edge"], expected_edge)
        self.assertEqual(result["ev"], expected_ev)

    def test_analyzer_unaffected(self):
        result = analyze_bet(2.10, 55)
        self.assertAlmostEqual(result["edge"], 7.38, places=2)
        self.assertAlmostEqual(result["ev"], 15.5, places=1)


class TestAnalyzeMarketUsesOverroundRemoval(unittest.TestCase):
    """`analyze_market()` já recebe, por definição, o conjunto completo das
    odds de um mercado — passa a reutilizá-las como `market_odds`."""

    def test_1x2_market_edge_uses_fair_probability(self):
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        result = analyze_market(odds, 0.55)
        for outcome, odd in odds.items():
            expected = calculate_edge(0.55, odd, market_odds=odds)
            self.assertEqual(result[outcome]["edge"], expected)

    def test_over_under_market_edge_uses_fair_probability(self):
        odds = {"OVER_2.5": 1.90, "UNDER_2.5": 1.90}
        result = analyze_market(odds, 0.55)
        for outcome, odd in odds.items():
            expected = calculate_edge(0.55, odd, market_odds=odds)
            self.assertEqual(result[outcome]["edge"], expected)

    def test_btts_market_edge_uses_fair_probability(self):
        odds = {"YES": 1.80, "NO": 2.00}
        result = analyze_market(odds, 0.55)
        for outcome, odd in odds.items():
            expected = calculate_edge(0.55, odd, market_odds=odds)
            self.assertEqual(result[outcome]["edge"], expected)

    def test_single_outcome_market_falls_back_to_old_behaviour(self):
        # Um único outcome -> insuficiente para remover overround ->
        # comportamento idêntico ao anterior a esta melhoria.
        odds = {"HOME": 2.10}
        result = analyze_market(odds, 0.55)
        expected = calculate_edge(0.55, 2.10)
        self.assertEqual(result["HOME"]["edge"], expected)

    def test_ev_field_still_uses_actual_house_odd_not_fair_odd(self):
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        result = analyze_market(odds, 0.55)
        for outcome, odd in odds.items():
            self.assertEqual(result[outcome]["ev"], calculate_ev(0.55, odd))

    def test_market_probability_field_unchanged_still_raw_implied(self):
        # Retrocompatibilidade: o campo "market_probability" continua a
        # ser a probabilidade implícita simples (com margem) — só o "edge"
        # passou a usar a probabilidade fair.
        odds = {"HOME": 1.65, "DRAW": 4.20, "AWAY": 4.60}
        result = analyze_market(odds, 0.55)
        for outcome, odd in odds.items():
            self.assertEqual(result[outcome]["market_probability"], implied_probability(odd))


class TestCliPredict1X2UsesOverroundRemoval(unittest.TestCase):
    """src/cli/predict.py::run_predict() já tem, por jogo, as 3 odds
    HOME/DRAW/AWAY disponíveis — passa a reutilizá-las como market_odds."""

    def test_each_market_edge_matches_fair_probability_computation(self):
        import unittest.mock as mock
        from src.models.match import Match
        import src.cli.predict as predict_module

        match = Match(
            home="Benfica",
            away="Porto",
            odds={"HOME": 1.80, "DRAW": 3.50, "AWAY": 4.50},
            probability=55,
            league="Liga Portugal",
            h2h_matches=6,
            dixon_coles_probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
        )

        market_odds_1x2 = {"HOME": 1.80, "DRAW": 3.50, "AWAY": 4.50}
        dc_probs = {"HOME": 0.55, "DRAW": 0.25, "AWAY": 0.20}

        with mock.patch.object(predict_module, "EventCollector") as MockCollector, \
             mock.patch.object(predict_module, "create_ranking") as mock_create_ranking:

            MockCollector.return_value.get_matches.return_value = [match]
            mock_create_ranking.return_value = {"value_bets": [], "watchlist": []}

            predict_module.run_predict()

            results = mock_create_ranking.call_args[0][0]

        by_market = {r["market"]: r for r in results}

        for market_name, odd in market_odds_1x2.items():
            expected_edge = round(
                calculate_edge(dc_probs[market_name], odd, market_odds=market_odds_1x2) * 100,
                2,
            )
            self.assertEqual(by_market[market_name]["edge"], expected_edge)


if __name__ == "__main__":
    unittest.main()
