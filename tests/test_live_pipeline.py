"""
Testes unitários da origem do λ (lambda) usado pela simulação Monte Carlo
em `src.engine.live_pipeline.LivePipeline`.

Cobrem:
  - o λ_home dinâmico (`calculate_dynamic_lambda`) é efetivamente o valor
    passado ao `MonteCarloSimulator.run_match_simulation`;
  - fallback seguro quando os dados ao vivo estão ausentes/inválidos
    (nunca deve levantar excepção nem deixar a simulação falhar);
  - `LivePipeline.evaluate()` devolve o λ realmente usado (`analysis["lambda"]`)
    e este é consistente com o `calculate_dynamic_lambda` chamado com o
    mesmo `live_result`;
  - compatibilidade: para o mesmo `live_result`, o λ_home dinâmico não
    é o valor fixo legado (1.6), confirmando que já não está "chumbado".
"""

import unittest

from src.engine.live_pipeline import (
    LivePipeline,
    FALLBACK_LAMBDA_HOME,
    FALLBACK_LAMBDA_AWAY,
)
from src.models.live_state import LiveMatchState


class FakeMatchProvider:
    def __init__(self, match_state: LiveMatchState):
        self._match_state = match_state

    def get_live_match(self, match_id):
        return self._match_state


class FakeOddsProvider:
    def get_live_odds(self, match_id):
        return {"odds": {"over_15_goals": 1.85}}


def make_pipeline(match_state: LiveMatchState) -> LivePipeline:
    return LivePipeline(
        match_provider=FakeMatchProvider(match_state),
        odds_provider=FakeOddsProvider(),
    )


class TestCalculateDynamicLambda(unittest.TestCase):

    def setUp(self):
        self.pipeline = LivePipeline(
            match_provider=FakeMatchProvider(LiveMatchState()),
            odds_provider=FakeOddsProvider(),
        )

    def test_dynamic_value_reflects_live_inputs(self):
        low_pressure = self.pipeline.calculate_dynamic_lambda(
            {"estimated_xg_10m": 0.1, "pressure": 10.0}
        )
        high_pressure = self.pipeline.calculate_dynamic_lambda(
            {"estimated_xg_10m": 1.5, "pressure": 90.0}
        )

        self.assertNotEqual(low_pressure, high_pressure)
        self.assertGreater(high_pressure, low_pressure)

    def test_matches_known_formula(self):
        result = self.pipeline.calculate_dynamic_lambda(
            {"estimated_xg_10m": 1.0, "pressure": 50.0}
        )
        expected = round(min(1.20 + 1.0 * 0.30 + (50.0 / 100) * 0.60, 4.0), 2)
        self.assertEqual(result, expected)

    def test_result_is_clamped_to_4_0(self):
        result = self.pipeline.calculate_dynamic_lambda(
            {"estimated_xg_10m": 999.0, "pressure": 100.0}
        )
        self.assertLessEqual(result, 4.0)

    def test_fallback_when_keys_missing(self):
        result = self.pipeline.calculate_dynamic_lambda({})
        self.assertEqual(result, FALLBACK_LAMBDA_HOME)

    def test_fallback_when_values_not_numeric(self):
        result = self.pipeline.calculate_dynamic_lambda(
            {"estimated_xg_10m": None, "pressure": "n/a"}
        )
        self.assertEqual(result, FALLBACK_LAMBDA_HOME)

    def test_fallback_never_raises(self):
        try:
            result = self.pipeline.calculate_dynamic_lambda(None)
        except Exception as exc:  # pragma: no cover - falha se levantar
            self.fail(f"calculate_dynamic_lambda levantou excepção: {exc}")
        self.assertEqual(result, FALLBACK_LAMBDA_HOME)


class TestEvaluateUsesDynamicLambda(unittest.TestCase):

    def test_evaluate_reports_dynamic_lambda_home(self):
        match_state = LiveMatchState(
            minute=60,
            dangerous_attacks_10m=18,
            shots_on_target_10m=5,
            shots_10m=12,
            corners_10m=4,
            previous_pressure=60,
        )
        pipeline = make_pipeline(match_state)

        live_result = pipeline.live_engine.predict_next_goal_probability(
            match_state
        )
        expected_lambda_home = pipeline.calculate_dynamic_lambda(live_result)

        analysis = pipeline.evaluate(match_id=1)

        self.assertEqual(analysis["lambda"]["home"], expected_lambda_home)
        # Já não é o valor fixo legado usado antes desta alteração.
        self.assertNotEqual(analysis["lambda"]["home"], 1.6)

    def test_evaluate_reports_dynamic_lambda_away(self):
        match_state = LiveMatchState(
            minute=30,
            away_conceded_xg_last5=2.4,
        )
        pipeline = make_pipeline(match_state)

        live_result = pipeline.live_engine.predict_next_goal_probability(
            match_state
        )
        expected_lambda_away = pipeline.calculate_dynamic_lambda(
            {
                "estimated_xg_10m": match_state.away_conceded_xg_last5,
                "pressure": live_result.get("pressure"),
            }
        )

        analysis = pipeline.evaluate(match_id=1)

        self.assertEqual(analysis["lambda"]["away"], expected_lambda_away)
        # Já não é o valor fixo legado usado antes desta alteração.
        self.assertNotEqual(analysis["lambda"]["away"], FALLBACK_LAMBDA_AWAY)

    def test_evaluate_lambda_away_varies_with_away_metrics(self):
        low = make_pipeline(
            LiveMatchState(minute=30, away_conceded_xg_last5=0.3)
        ).evaluate(match_id=1)
        high = make_pipeline(
            LiveMatchState(minute=30, away_conceded_xg_last5=3.0)
        ).evaluate(match_id=1)

        self.assertNotEqual(low["lambda"]["away"], high["lambda"]["away"])

    def test_evaluate_uses_real_match_score(self):
        match_state = LiveMatchState(minute=70, home_score=2, away_score=1)
        pipeline = make_pipeline(match_state)

        analysis = pipeline.evaluate(match_id=1)

        # Com 3 golos já marcados no tempo real do jogo, over_15 tem de ser
        # 100% mesmo que não se marque mais nenhum golo no tempo restante.
        self.assertEqual(analysis["simulation"]["over_15"], 100.0)
        self.assertGreaterEqual(analysis["simulation"]["expected_home_goals"], 2.0)
        self.assertGreaterEqual(analysis["simulation"]["expected_away_goals"], 1.0)

    def test_red_card_reduces_away_lambda_only(self):
        base_state = LiveMatchState(minute=40, red_cards=0)
        red_card_state = LiveMatchState(minute=40, red_cards=1)

        base = make_pipeline(base_state).evaluate(match_id=7)
        with_red_card = make_pipeline(red_card_state).evaluate(match_id=7)

        self.assertEqual(base["lambda"]["home"], with_red_card["lambda"]["home"])
        self.assertLess(with_red_card["lambda"]["away"], base["lambda"]["away"])

    def test_evaluate_is_reproducible_for_same_match_id_and_minute(self):
        match_state = LiveMatchState(
            minute=55,
            dangerous_attacks_10m=10,
            shots_10m=6,
            shots_on_target_10m=3,
            corners_10m=2,
        )

        first = make_pipeline(match_state).evaluate(match_id=42)
        second = make_pipeline(match_state).evaluate(match_id=42)

        self.assertEqual(first["simulation"], second["simulation"])

    def test_evaluate_never_fails_even_with_bare_match_state(self):
        # Estado mínimo, sem qualquer sinal de pressão ao vivo.
        pipeline = make_pipeline(LiveMatchState())
        try:
            analysis = pipeline.evaluate(match_id=1)
        except Exception as exc:  # pragma: no cover - falha se levantar
            self.fail(f"evaluate() levantou excepção: {exc}")

        self.assertIn("simulation", analysis)
        self.assertIn("lambda", analysis)

    def test_simulation_result_shape_unchanged(self):
        pipeline = make_pipeline(LiveMatchState(minute=45))
        analysis = pipeline.evaluate(match_id=1)

        sim = analysis["simulation"]
        for key in (
            "over_15",
            "over_25",
            "btts",
            "expected_home_goals",
            "expected_away_goals",
        ):
            self.assertIn(key, sim)


if __name__ == "__main__":
    unittest.main()
