"""Testes de `APIMatchProvider.get_live_match()`.

Cobrem a correção da melhoria #1 da auditoria técnica: o provider deixa de
forçar `dangerous_attacks_10m`, `shots_10m`, `shots_on_target_10m`,
`corners_10m` e `possession` a valores hard-coded (0 / 50.0) e passa a
extrair estes campos dos dados reais já devolvidos por `StatsProvider`
(`/events/{id}/stats/`), sem qualquer nova chamada HTTP.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.live.providers.api_match_provider import APIMatchProvider


def _mock_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestGetLiveMatchUsesRealStats(unittest.TestCase):

    @patch("src.live.providers.api_match_provider.get_with_retry")
    @patch("src.live.providers.stats_provider.get_with_retry")
    @patch("src.live.providers.incidents_provider.get_with_retry")
    def test_real_stats_fields_replace_hardcoded_values(
        self, mock_incidents_get, mock_stats_get, mock_event_get
    ):
        mock_event_get.return_value = _mock_response({
            "current_minute": 60,
            "home_score": 1,
            "away_score": 0,
        })

        mock_stats_get.return_value = _mock_response({
            "stats": {
                "home": {
                    "dangerous_attack": 14,
                    "shots_total": 8,
                    "shots_on_target": 3,
                    "corners": 5,
                    "possession": 58,
                    "xg": {"actual": 1.4},
                },
                "away": {
                    "dangerous_attack": 6,
                    "shots_total": 4,
                    "shots_on_target": 1,
                    "corners": 2,
                    "possession": 42,
                    "xg": {"actual": 0.6},
                },
            }
        })

        mock_incidents_get.return_value = _mock_response({"incidents": []})

        provider = APIMatchProvider()
        match_state = provider.get_live_match(match_id=123)

        # Nenhuma chamada HTTP extra além das 3 já existentes (event, stats, incidents).
        self.assertEqual(mock_event_get.call_count, 1)
        self.assertEqual(mock_stats_get.call_count, 1)
        self.assertEqual(mock_incidents_get.call_count, 1)

        self.assertEqual(match_state.dangerous_attacks_10m, 20)
        self.assertEqual(match_state.shots_10m, 12)
        self.assertEqual(match_state.shots_on_target_10m, 4)
        self.assertEqual(match_state.corners_10m, 7)
        self.assertEqual(match_state.possession, 58.0)

        # Já não são os valores hard-coded antigos.
        self.assertNotEqual(match_state.dangerous_attacks_10m, 0)
        self.assertNotEqual(match_state.shots_10m, 0)
        self.assertNotEqual(match_state.shots_on_target_10m, 0)
        self.assertNotEqual(match_state.corners_10m, 0)
        self.assertNotEqual(match_state.possession, 50.0)

    @patch("src.live.providers.api_match_provider.get_with_retry")
    @patch("src.live.providers.stats_provider.get_with_retry")
    @patch("src.live.providers.incidents_provider.get_with_retry")
    def test_missing_stats_falls_back_safely(
        self, mock_incidents_get, mock_stats_get, mock_event_get
    ):
        mock_event_get.return_value = _mock_response({
            "current_minute": 10,
            "home_score": 0,
            "away_score": 0,
        })

        mock_stats_get.return_value = _mock_response({"stats": {}})
        mock_incidents_get.return_value = _mock_response({"incidents": []})

        provider = APIMatchProvider()
        match_state = provider.get_live_match(match_id=456)

        self.assertEqual(match_state.dangerous_attacks_10m, 0)
        self.assertEqual(match_state.shots_10m, 0)
        self.assertEqual(match_state.shots_on_target_10m, 0)
        self.assertEqual(match_state.corners_10m, 0)
        self.assertEqual(match_state.possession, 50.0)


if __name__ == "__main__":
    unittest.main()
