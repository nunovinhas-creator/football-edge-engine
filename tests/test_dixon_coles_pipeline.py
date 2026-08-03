"""
Testes de integração da entrada em produção do modelo Dixon-Coles já
existente (`src/engine/dixon_coles.py`) na pipeline `main.py predict`.

Contexto (ver docs/AUDIT_MATEMATICA.md, secção "Dixon-Coles entra em
produção"): antes desta integração, `dixon_coles_simulate_match()` e
`evaluate_match_value()` nunca eram chamados por nenhum entry point de
produção — apenas por testes. `src/cli/predict.py` usava a mesma
probabilidade heurística de H2H (`match.probability`) para os três
mercados (HOME/DRAW/AWAY), o que era matematicamente incorreto (a mesma
probabilidade não pode ser simultaneamente P(home) e P(away)).

Estes testes cobrem:
  - `src/engine/dixon_coles.py` (tau, dixon_coles_simulate_match,
    calculate_fractional_kelly) permanece exatamente como estava —
    caracterização de comportamento, sem alterações à fórmula;
  - `src/engine/value.py::evaluate_match_value` produz exatamente os
    mesmos números depois do refactor que extraiu
    `_market_probabilities_from_matrix` (não houve mudança de
    comportamento, só remoção de duplicação);
  - `src/engine/value.py::estimate_pregame_probabilities` (novo) devolve
    probabilidades 1X2 coerentes com a mesma matriz Dixon-Coles;
  - `src/engine/pregame_lambda.py::estimate_pregame_lambdas` (novo
    adaptador de inputs, não um novo modelo) nunca falha e reage aos
    dados de H2H disponíveis;
  - `src/collector/client.py::EventCollector.get_matches()` passa a
    anexar `dixon_coles_probabilities` a cada `Match`;
  - `src/cli/predict.py::run_predict()` usa essas probabilidades por
    mercado (já não a mesma probabilidade para HOME/DRAW/AWAY).
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.engine.dixon_coles import (
    tau,
    dixon_coles_simulate_match,
    calculate_fractional_kelly,
)
from src.engine.value import (
    evaluate_match_value,
    estimate_pregame_probabilities,
    _market_probabilities_from_matrix,
)
from src.engine.pregame_lambda import (
    estimate_pregame_lambdas,
    DEFAULT_AVG_TOTAL_GOALS,
    MIN_LAMBDA,
)
from src.models.match import Match


class TestDixonColesModelUnchanged(unittest.TestCase):
    """Caracterização do núcleo Dixon-Coles (tau, simulação, Kelly) —
    confirma que esta integração não alterou a fórmula existente."""

    def test_tau_known_values(self):
        self.assertAlmostEqual(tau(0, 0, 1.5, 1.1, -0.05), 1.0 - (1.5 * 1.1 * -0.05))
        self.assertAlmostEqual(tau(1, 0, 1.5, 1.1, -0.05), 1.0 + (1.1 * -0.05))
        self.assertAlmostEqual(tau(0, 1, 1.5, 1.1, -0.05), 1.0 + (1.5 * -0.05))
        self.assertAlmostEqual(tau(1, 1, 1.5, 1.1, -0.05), 1.0 - (-0.05))
        self.assertEqual(tau(3, 2, 1.5, 1.1, -0.05), 1.0)

    def test_matrix_sums_to_one(self):
        matrix = dixon_coles_simulate_match(1.5, 1.1)
        self.assertAlmostEqual(float(np.sum(matrix)), 1.0, places=9)

    def test_matrix_shape_uses_default_max_goals(self):
        matrix = dixon_coles_simulate_match(1.5, 1.1)
        self.assertEqual(matrix.shape, (9, 9))

    def test_fractional_kelly_known_value(self):
        # p=0.55, odd=2.10 -> kelly_full=(1.10*0.55-0.45)/1.10=0.14545...
        # fractional (1/4) = 0.036363..., abaixo do cap de 2% -> cap aplica-se
        stake = calculate_fractional_kelly(0.55, 2.10)
        self.assertEqual(stake, 0.02)

    def test_fractional_kelly_negative_edge_is_zero(self):
        self.assertEqual(calculate_fractional_kelly(0.3, 1.5), 0.0)


class TestEvaluateMatchValueRegression(unittest.TestCase):
    """evaluate_match_value() não pode ter mudado de comportamento depois
    de extrair _market_probabilities_from_matrix()."""

    def test_probabilities_sum_to_one(self):
        result = evaluate_match_value(1.6, 1.1, 1.90, 3.40, 4.20)
        total = result["home"]["prob"] + result["draw"]["prob"] + result["away"]["prob"]
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_matches_manual_matrix_computation(self):
        lambda_home, mu_away = 1.6, 1.1
        matrix = dixon_coles_simulate_match(lambda_home, mu_away)
        expected_home = float(np.sum(np.tril(matrix, -1)))
        expected_draw = float(np.trace(matrix))
        expected_away = float(np.sum(np.triu(matrix, 1)))

        result = evaluate_match_value(lambda_home, mu_away, 1.90, 3.40, 4.20)

        self.assertAlmostEqual(result["home"]["prob"], expected_home, places=9)
        self.assertAlmostEqual(result["draw"]["prob"], expected_draw, places=9)
        self.assertAlmostEqual(result["away"]["prob"], expected_away, places=9)

    def test_ev_and_stake_untouched(self):
        result = evaluate_match_value(1.6, 1.1, 1.90, 3.40, 4.20)
        p_home = result["home"]["prob"]
        self.assertAlmostEqual(result["home"]["ev"], (p_home * 1.90) - 1.0, places=9)
        self.assertEqual(
            result["home"]["stake_pct"],
            calculate_fractional_kelly(p_home, 1.90),
        )


class TestEstimatePregameProbabilities(unittest.TestCase):
    """Novo helper de src/engine/value.py: probabilidades sem exigir odds."""

    def test_matches_market_probabilities_from_same_matrix(self):
        matrix = dixon_coles_simulate_match(1.6, 1.1)
        expected = _market_probabilities_from_matrix(matrix)

        result = estimate_pregame_probabilities(1.6, 1.1)

        self.assertAlmostEqual(result["home"], expected[0], places=9)
        self.assertAlmostEqual(result["draw"], expected[1], places=9)
        self.assertAlmostEqual(result["away"], expected[2], places=9)

    def test_matches_evaluate_match_value_probabilities(self):
        # Mesma matriz subjacente -> mesmas probabilidades, com ou sem odds.
        full = evaluate_match_value(1.6, 1.1, 1.90, 3.40, 4.20)
        probs_only = estimate_pregame_probabilities(1.6, 1.1)

        self.assertAlmostEqual(probs_only["home"], full["home"]["prob"], places=9)
        self.assertAlmostEqual(probs_only["draw"], full["draw"]["prob"], places=9)
        self.assertAlmostEqual(probs_only["away"], full["away"]["prob"], places=9)

    def test_probabilities_sum_to_one(self):
        result = estimate_pregame_probabilities(2.2, 0.7)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=9)

    def test_stronger_home_lambda_increases_home_probability(self):
        weak_home = estimate_pregame_probabilities(1.0, 1.0)
        strong_home = estimate_pregame_probabilities(2.5, 1.0)
        self.assertGreater(strong_home["home"], weak_home["home"])


class TestEstimatePregameLambdas(unittest.TestCase):
    """Adaptador de inputs (src/engine/pregame_lambda.py) — não é um novo
    modelo, apenas fornece lambda_home/mu_away ao Dixon-Coles existente."""

    def test_none_h2h_uses_default_and_never_raises(self):
        lambda_home, mu_away = estimate_pregame_lambdas(None)
        self.assertGreater(lambda_home, 0)
        self.assertGreater(mu_away, 0)
        self.assertAlmostEqual(lambda_home + mu_away, DEFAULT_AVG_TOTAL_GOALS, places=2)

    def test_empty_dict_behaves_like_none(self):
        result_empty = estimate_pregame_lambdas({})
        result_none = estimate_pregame_lambdas(None)
        self.assertEqual(result_empty, result_none)

    def test_uses_avg_total_goals_from_h2h(self):
        lambda_home, mu_away = estimate_pregame_lambdas({"avg_total_goals": 4.0})
        self.assertAlmostEqual(lambda_home + mu_away, 4.0, places=2)

    def test_home_advantage_tilts_split_towards_home(self):
        lambda_home, mu_away = estimate_pregame_lambdas({
            "avg_total_goals": 2.5,
            "home_win_rate": 50,
            "away_win_rate": 50,
        })
        self.assertGreater(lambda_home, mu_away)

    def test_strong_away_h2h_record_still_respects_cap(self):
        lambda_home, mu_away = estimate_pregame_lambdas({
            "avg_total_goals": 2.5,
            "home_win_rate": 0,
            "away_win_rate": 100,
        })
        # tilt está capado (MAX_STRENGTH_TILT), não pode inverter para uma
        # repartição absurda (ex. away com 0 golos esperados).
        self.assertGreaterEqual(mu_away, MIN_LAMBDA)
        self.assertGreaterEqual(lambda_home, MIN_LAMBDA)

    def test_zero_avg_total_goals_falls_back_to_default(self):
        lambda_home, mu_away = estimate_pregame_lambdas({"avg_total_goals": 0})
        self.assertAlmostEqual(lambda_home + mu_away, DEFAULT_AVG_TOTAL_GOALS, places=2)

    def test_never_returns_lambda_below_minimum(self):
        lambda_home, mu_away = estimate_pregame_lambdas({"avg_total_goals": 0.1})
        self.assertGreaterEqual(lambda_home, MIN_LAMBDA)
        self.assertGreaterEqual(mu_away, MIN_LAMBDA)


class TestEventCollectorAttachesDixonColesProbabilities(unittest.TestCase):
    """src/collector/client.py::EventCollector.get_matches() passa a
    calcular e anexar as probabilidades Dixon-Coles a cada Match."""

    def _make_event(self, event_id=1, h2h=None):
        return {
            "id": event_id,
            "home_team": "Benfica",
            "away_team": "Porto",
            "league_id": 42,
            "head_to_head": h2h,
        }

    def test_match_gets_dixon_coles_probabilities(self):
        from src.collector.client import EventCollector

        with patch("src.collector.client.BzzoiroClient") as MockClient, \
             patch("src.collector.client.OddsCollector") as MockOdds:

            MockClient.return_value.get.return_value = {
                "results": [self._make_event(h2h={
                    "total_matches": 8,
                    "avg_total_goals": 3.0,
                    "home_win_rate": 60,
                    "away_win_rate": 20,
                })]
            }
            MockOdds.return_value.get_event_odds.return_value = {
                "HOME": 1.80, "DRAW": 3.50, "AWAY": 4.50
            }

            collector = EventCollector()
            matches = collector.get_matches(1)

        self.assertEqual(len(matches), 1)
        match = matches[0]

        self.assertIsNotNone(match.dixon_coles_probabilities)
        for key in ("home", "draw", "away"):
            self.assertIn(key, match.dixon_coles_probabilities)

        total = sum(match.dixon_coles_probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=6)

        # golos esperados (lambda_home/mu_away) passam a ficar em xg_home/xg_away
        self.assertIsNotNone(match.xg_home)
        self.assertIsNotNone(match.xg_away)

        # A partir de src/engine/lambda_estimator.py (estimador por omissão
        # da pipeline desde a introdução do estimador estatisticamente mais
        # forte — ver docs/05_lambda_estimator.md), a soma xg_home+xg_away
        # já não é necessariamente igual a avg_total_goals: sem
        # "home_goals"/"away_goals"/"recent_matches" no H2H, a repartição
        # inferida (Nível C, delegada a pregame_lambda.py) é encolhida
        # (shrinkage) para o prior de liga (2.5) proporcionalmente a
        # total_matches*0.5=4 "pseudo-jogos" contra SHRINKAGE_K=4 do prior —
        # ou seja, prior e amostra pesam o mesmo aqui. Valores exatos
        # cobertos em detalhe por tests/test_lambda_estimator.py.
        self.assertAlmostEqual(match.xg_home, 1.75, places=2)
        self.assertAlmostEqual(match.xg_away, 1.0, places=2)

    def test_match_gets_dixon_coles_probabilities_even_without_h2h(self):
        from src.collector.client import EventCollector

        with patch("src.collector.client.BzzoiroClient") as MockClient, \
             patch("src.collector.client.OddsCollector") as MockOdds:

            MockClient.return_value.get.return_value = {
                "results": [self._make_event(h2h=None)]
            }
            MockOdds.return_value.get_event_odds.return_value = {
                "HOME": 2.0, "DRAW": 3.2, "AWAY": 3.8
            }

            collector = EventCollector()
            matches = collector.get_matches(1)

        match = matches[0]
        self.assertIsNotNone(match.dixon_coles_probabilities)
        self.assertAlmostEqual(sum(match.dixon_coles_probabilities.values()), 1.0, places=6)


class TestRunPredictUsesDixonColesPerMarket(unittest.TestCase):
    """src/cli/predict.py::run_predict() já não usa a mesma probabilidade
    para HOME/DRAW/AWAY — usa a probabilidade Dixon-Coles por mercado."""

    def _make_match(self):
        return Match(
            home="Benfica",
            away="Porto",
            odds={"HOME": 1.80, "DRAW": 3.50, "AWAY": 4.50},
            probability=55,  # heurística H2H antiga (agora só fallback)
            league="Liga Portugal",
            h2h_matches=6,
            dixon_coles_probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
        )

    def test_each_market_gets_its_own_model_probability(self):
        import src.cli.predict as predict_module

        with patch.object(predict_module, "EventCollector") as MockCollector, \
             patch.object(predict_module, "create_ranking") as mock_create_ranking:

            MockCollector.return_value.get_matches.return_value = [self._make_match()]
            mock_create_ranking.return_value = {"value_bets": [], "watchlist": []}

            predict_module.run_predict()

            results = mock_create_ranking.call_args[0][0]

        by_market = {r["market"]: r["model_probability"] for r in results}

        self.assertEqual(by_market["HOME"], 55.0)
        self.assertEqual(by_market["DRAW"], 25.0)
        self.assertEqual(by_market["AWAY"], 20.0)
        # Antes desta integração, os 3 valores eram idênticos (match.probability).
        self.assertEqual(len(set(by_market.values())), 3)

    def test_falls_back_to_h2h_heuristic_when_dixon_coles_missing(self):
        import src.cli.predict as predict_module

        match = Match(
            home="Sporting", away="Braga",
            odds={"HOME": 2.0, "DRAW": 3.3, "AWAY": 3.6},
            probability=60,
            dixon_coles_probabilities=None,
        )

        with patch.object(predict_module, "EventCollector") as MockCollector, \
             patch.object(predict_module, "create_ranking") as mock_create_ranking:

            MockCollector.return_value.get_matches.return_value = [match]
            mock_create_ranking.return_value = {"value_bets": [], "watchlist": []}

            predict_module.run_predict()

            results = mock_create_ranking.call_args[0][0]

        for r in results:
            self.assertEqual(r["model_probability"], 60.0)


if __name__ == "__main__":
    unittest.main()
