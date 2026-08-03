"""
Testes de integração do consumo do λ dinâmico por
`src.report.dashboard.render_live_dashboard`.

Cobrem:
  - o dashboard já não instancia/chama `MonteCarloSimulator` (não há
    segunda simulação com λ fixos 1.6/1.1 — ver docs/AUDIT_MATEMATICA.md,
    secção "Duas simulações desconectadas no mesmo output");
  - os valores mostrados/usados (via `DecisionEngine.evaluate_bet`) são
    exatamente os que vêm de `analysis["simulation"]`, produzidos pelo λ
    dinâmico de `LivePipeline`.
"""

import unittest
from unittest.mock import patch

from src.models.live_state import LiveMatchState
from src.report.dashboard import render_live_dashboard


def make_analysis(over_15_prob: float):
    return {
        "live": {
            "pressure": 42.0,
            "dominance_index": 55.0,
            "estimated_xg_10m": 0.8,
            "next_goal_probability": 48.0,
            "recommendation": "⚠️ WAIT (PRESSURE BUILDING)",
        },
        "lambda": {"home": 2.1, "away": 0.80},
        "simulation": {
            "over_15": over_15_prob,
            "over_25": 40.0,
            "btts": 55.0,
            "expected_home_goals": 1.4,
            "expected_away_goals": 0.9,
        },
    }


class TestDashboardConsumesPipelineSimulation(unittest.TestCase):

    def setUp(self):
        self.match_state = LiveMatchState(minute=55)

    def test_does_not_run_a_second_monte_carlo_simulation(self):
        analysis = make_analysis(over_15_prob=91.3)

        with patch(
            "src.engine.simulation.MonteCarloSimulator.run_match_simulation"
        ) as mock_run:
            render_live_dashboard(
                home_team="Home",
                away_team="Away",
                score="1-0",
                match_state=self.match_state,
                bookie_over15_odd=1.80,
                analysis=analysis,
            )
            mock_run.assert_not_called()

    def test_bet_decision_uses_pipeline_simulation_value(self):
        analysis = make_analysis(over_15_prob=91.3)

        with patch(
            "src.report.dashboard.DecisionEngine.evaluate_bet"
        ) as mock_evaluate_bet:
            mock_evaluate_bet.return_value = type(
                "BetRecommendation",
                (),
                {"edge_pct": 0.0, "kelly_stake_pct": 0.0, "action": "PASS"},
            )()

            render_live_dashboard(
                home_team="Home",
                away_team="Away",
                score="1-0",
                match_state=self.match_state,
                bookie_over15_odd=1.80,
                analysis=analysis,
            )

            mock_evaluate_bet.assert_called_once_with(
                "Over 1.5", 91.3, 1.80
            )

    def test_different_dynamic_lambda_changes_displayed_probability(self):
        low = make_analysis(over_15_prob=30.0)
        high = make_analysis(over_15_prob=95.0)

        fake_bet_rec = type(
            "BetRecommendation",
            (),
            {"edge_pct": 0.0, "kelly_stake_pct": 0.0, "action": "PASS"},
        )()

        captured = []
        with patch(
            "src.report.dashboard.DecisionEngine.evaluate_bet"
        ) as mock_evaluate_bet:
            def fake_evaluate_bet(market, prob, odd):
                captured.append(prob)
                return fake_bet_rec

            mock_evaluate_bet.side_effect = fake_evaluate_bet

            render_live_dashboard(
                home_team="Home", away_team="Away", score="0-0",
                match_state=self.match_state, bookie_over15_odd=1.80,
                analysis=low,
            )
            render_live_dashboard(
                home_team="Home", away_team="Away", score="0-0",
                match_state=self.match_state, bookie_over15_odd=1.80,
                analysis=high,
            )

        self.assertEqual(captured, [30.0, 95.0])


if __name__ == "__main__":
    unittest.main()
