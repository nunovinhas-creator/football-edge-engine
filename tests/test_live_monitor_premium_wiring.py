"""
Testes de integração da ligação entre `src.engine.live_monitor` (o
monitor automático executado pelo workflow `live_logger.yml`), o Alerta
Live Premium (`src.alerts.live_premium_alerts`) e o Goal Imminent
Detection (`src.alerts.goal_imminent_detector`).

Cobrem apenas a "cablagem" (wiring):
  - `run_live_pipeline()` constrói o MatchSnapshot real via
    `build_match_snapshot` e chama `LiveAlertMonitor.evaluate_and_maybe_alert`
    exatamente uma vez por jogo em direto com odds disponíveis;
  - `run_live_pipeline()` chama também
    `GoalImminentDetector.evaluate_and_maybe_alert` exatamente uma vez por
    jogo, com o MESMO MatchSnapshot já usado pelo Alerta Live Premium
    (nenhum dos dois recalcula nada — ver
    `tests/test_goal_imminent_detector.py` para os 12 critérios em si);
  - a odd real já obtida do `odds_provider` é reutilizada (via
    `match_data["live_odd_over"]`) em vez de cair no odd de fallback;
  - `LiveAlertMonitor.sync_active_matches` é chamado com os match_ids dos
    jogos atualmente em direto (para limpar jogos terminados).

Não recalcula nem reavalia os critérios de nenhum dos dois alertas aqui
(isso já está coberto por `tests/test_live_premium_alerts.py` e
`tests/test_goal_imminent_detector.py`) — usa um `LiveMLPredictor` falso
(sem carregar nenhum modelo real) só para manter o teste rápido e
determinístico.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.model.ml_predictor import MLPredictionResult


class FakeMLPredictor:
    def predict(self, match_state, live_odd_over):
        return MLPredictionResult(
            goal_probability=72.0,
            confidence_score=80.0,
            model_used="Fake (teste de wiring)",
        )


def _fake_live_event(match_id=42):
    return {
        "id": match_id,
        "home_team": "Benfica",
        "away_team": "Sporting",
        "current_minute": 55,
        "home_score": 1,
        "away_score": 0,
        "statistics": {
            "home": {"dangerous_attacks": 10, "shots_on_target": 4, "corners": 3},
            "away": {"dangerous_attacks": 4, "shots_on_target": 1, "corners": 1},
        },
    }


class TestLiveMonitorCallsPremiumAlertMonitor(unittest.TestCase):

    @patch("src.engine.live_monitor.GoalImminentDetector")
    @patch("src.engine.live_monitor.LiveAlertMonitor")
    @patch("src.engine.live_monitor.LiveMLPredictor")
    @patch("src.engine.live_monitor.APIOddsProvider")
    @patch("src.engine.live_monitor.BSDLiveFetcher")
    @patch("src.engine.live_monitor.notify_if_value", return_value=False)
    @patch("src.engine.live_monitor.log_snapshot")
    @patch("src.engine.live_monitor.init_db")
    def test_evaluates_premium_alert_with_real_odd_and_snapshot(
        self,
        mock_init_db,
        mock_log_snapshot,
        mock_notify_if_value,
        MockFetcherClass,
        MockOddsProviderClass,
        MockMLPredictorClass,
        MockAlertMonitorClass,
        MockGoalImminentDetectorClass,
    ):
        from src.engine.live_monitor import run_live_pipeline

        event = _fake_live_event(match_id=42)

        mock_fetcher = MockFetcherClass.return_value
        mock_fetcher.get_live_events.return_value = [event]
        # Usa o parser real (não mockado) para produzir o match_data.
        from src.api.live_fetcher import BSDLiveFetcher
        mock_fetcher.parse_live_metrics_for_engine.side_effect = (
            BSDLiveFetcher.parse_live_metrics_for_engine.__get__(mock_fetcher)
        )

        mock_odds_provider = MockOddsProviderClass.return_value
        mock_odds_provider.get_live_odds.return_value = {"odds": {"over_15_goals": 1.95}}

        MockMLPredictorClass.return_value = FakeMLPredictor()

        mock_alert_monitor = MockAlertMonitorClass.return_value
        mock_alert_monitor.evaluate_and_maybe_alert.return_value = MagicMock(sent=False)

        mock_goal_imminent_detector = MockGoalImminentDetectorClass.return_value
        mock_goal_imminent_detector.evaluate_and_maybe_alert.return_value = MagicMock(sent=False)

        run_live_pipeline()

        mock_alert_monitor.sync_active_matches.assert_called_once()
        synced_ids = list(mock_alert_monitor.sync_active_matches.call_args.args[0])
        self.assertEqual(synced_ids, [42])

        mock_alert_monitor.evaluate_and_maybe_alert.assert_called_once()
        (snapshot_arg,), _ = mock_alert_monitor.evaluate_and_maybe_alert.call_args

        self.assertEqual(snapshot_arg["match_id"], 42)
        self.assertEqual(snapshot_arg["value"]["bookie_odd"], 1.95)
        self.assertEqual(snapshot_arg["card"]["home_team"], "Benfica")
        self.assertEqual(snapshot_arg["card"]["away_team"], "Sporting")

        # Goal Imminent Detection é avaliado exatamente uma vez, com o
        # MESMO objeto de snapshot já usado pelo Alerta Live Premium —
        # nenhum dos dois recalcula nada.
        mock_goal_imminent_detector.evaluate_and_maybe_alert.assert_called_once()
        (goal_imminent_snapshot_arg,), _ = mock_goal_imminent_detector.evaluate_and_maybe_alert.call_args
        self.assertIs(goal_imminent_snapshot_arg, snapshot_arg)

    @patch("src.engine.live_monitor.GoalImminentDetector")
    @patch("src.engine.live_monitor.LiveAlertMonitor")
    @patch("src.engine.live_monitor.LiveMLPredictor")
    @patch("src.engine.live_monitor.APIOddsProvider")
    @patch("src.engine.live_monitor.BSDLiveFetcher")
    @patch("src.engine.live_monitor.init_db")
    def test_no_odds_provider_skips_premium_alert_without_crashing(
        self,
        mock_init_db,
        MockFetcherClass,
        MockOddsProviderClass,
        MockMLPredictorClass,
        MockAlertMonitorClass,
        MockGoalImminentDetectorClass,
    ):
        from src.engine.live_monitor import run_live_pipeline

        event = _fake_live_event(match_id=7)
        mock_fetcher = MockFetcherClass.return_value
        mock_fetcher.get_live_events.return_value = [event]
        from src.api.live_fetcher import BSDLiveFetcher
        mock_fetcher.parse_live_metrics_for_engine.side_effect = (
            BSDLiveFetcher.parse_live_metrics_for_engine.__get__(mock_fetcher)
        )

        MockOddsProviderClass.side_effect = Exception("sem chave de odds configurada")
        MockMLPredictorClass.return_value = FakeMLPredictor()
        mock_alert_monitor = MockAlertMonitorClass.return_value
        mock_goal_imminent_detector = MockGoalImminentDetectorClass.return_value

        run_live_pipeline()

        mock_alert_monitor.evaluate_and_maybe_alert.assert_not_called()
        mock_alert_monitor.sync_active_matches.assert_called_once()
        mock_goal_imminent_detector.evaluate_and_maybe_alert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
