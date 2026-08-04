"""Testes do boletim Telegram gerado por src/cli/predict.py.

Cobre apenas a construção/envio da mensagem (camada de notificações),
sem tocar em edge/EV/decisão/thresholds. Usa mocks — nunca envia
mensagens reais.
"""

import unittest
from datetime import datetime
from unittest.mock import patch

from src.cli.predict import decision_alert_label, format_bet_alert, send_telegram_bulletin


class TestDecisionAlertLabel(unittest.TestCase):

    def test_bet_maps_to_apostar(self):
        self.assertEqual(decision_alert_label("BET 🔥"), "🟢 APOSTAR")

    def test_wait_maps_to_aguardar(self):
        self.assertEqual(decision_alert_label("WAIT ⚠️"), "🟡 AGUARDAR")

    def test_pass_maps_to_passar(self):
        self.assertEqual(decision_alert_label("PASS ❄️"), "⚪ PASSAR")


class TestFormatBetAlert(unittest.TestCase):

    def test_includes_all_expected_fields(self):
        bet = {
            "match": "Liverpool vs Arsenal",
            "market": "Over 1.5",
            "model_probability": 83,
            "odd": 1.72,
            "edge": 9.4,
            "ev": 15.0,
            "stake": 1.8,
            "decision": "BET 🔥",
        }
        now = datetime(2026, 8, 4, 19, 42)

        text = format_bet_alert(bet, now)

        self.assertIn("🟢 APOSTAR", text)
        self.assertIn("Liverpool vs Arsenal", text)
        self.assertIn("Over 1.5", text)
        self.assertIn("83%", text)
        self.assertIn("1.72", text)
        self.assertIn("+9.4%", text)
        self.assertIn("+15.0%", text)
        self.assertIn("1.8%", text)
        self.assertIn("19:42", text)


class TestSendTelegramBulletin(unittest.TestCase):

    @patch("src.cli.predict.send_telegram_alert")
    def test_sends_info_message_when_no_value_bets(self, mock_send):
        send_telegram_bulletin([])

        mock_send.assert_called_once()
        self.assertIn("Nenhuma oportunidade", mock_send.call_args.args[0])

    @patch("src.cli.predict.send_telegram_alert")
    def test_sends_one_alert_per_value_bet(self, mock_send):
        bets = [
            {"match": "A vs B", "market": "HOME", "model_probability": 60,
             "odd": 2.0, "edge": 6.0, "ev": 12.0, "stake": 1.0, "decision": "BET 🔥"},
            {"match": "C vs D", "market": "AWAY", "model_probability": 55,
             "odd": 2.1, "edge": 7.0, "ev": 13.0, "stake": 1.2, "decision": "BET 🔥"},
        ]

        send_telegram_bulletin(bets)

        self.assertEqual(mock_send.call_count, 2)

    @patch("src.cli.predict.send_telegram_alert")
    def test_caps_alerts_at_max_alerts(self, mock_send):
        bets = [
            {"match": f"Team{i} vs Team{i+1}", "market": "HOME", "model_probability": 60,
             "odd": 2.0, "edge": 6.0, "ev": 12.0, "stake": 1.0, "decision": "BET 🔥"}
            for i in range(10)
        ]

        send_telegram_bulletin(bets, max_alerts=3)

        self.assertEqual(mock_send.call_count, 3)


if __name__ == "__main__":
    unittest.main()
