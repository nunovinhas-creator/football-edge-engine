"""Testes do notifier Telegram central (src/utils/telegram_notifier.py).

Usa mocks em todos os casos — nunca contacta a API real do Telegram.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.utils.telegram_notifier import BRAND_PREFIX, send_telegram_alert


def _fake_response(status_code=200, text="OK"):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


class TestMessageBrandingAndPayload(unittest.TestCase):
    """Fase 3: toda a mensagem enviada tem de começar pelo prefixo
    obrigatório e a chamada à API tem de usar o token/chat_id corretos."""

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"})
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_message_has_brand_prefix(self, mock_post):
        mock_post.return_value = _fake_response(200)

        result = send_telegram_alert("🟢 APOSTAR\n\nLiverpool vs Arsenal")

        self.assertTrue(result)
        sent_text = mock_post.call_args.kwargs["json"]["text"]
        self.assertTrue(sent_text.startswith(BRAND_PREFIX))
        self.assertIn("Liverpool vs Arsenal", sent_text)

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"})
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_calls_correct_telegram_api_url_and_chat_id(self, mock_post):
        mock_post.return_value = _fake_response(200)

        send_telegram_alert("mensagem de teste")

        called_url = mock_post.call_args.args[0]
        called_payload = mock_post.call_args.kwargs["json"]

        self.assertEqual(called_url, "https://api.telegram.org/bottok123/sendMessage")
        self.assertEqual(called_payload["chat_id"], "chat456")
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 15)


class TestErrorHandling(unittest.TestCase):
    """Fase 4: nenhum erro pode ser engolido em silêncio."""

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"})
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_http_error_returns_false_and_is_not_silent(self, mock_post):
        mock_post.return_value = _fake_response(400, text='{"ok":false,"description":"Bad Request"}')

        with self.assertLogs("src.utils.telegram_notifier", level="ERROR") as log_ctx:
            result = send_telegram_alert("mensagem")

        self.assertFalse(result)
        self.assertTrue(any("400" in line for line in log_ctx.output))

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"})
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_timeout_is_handled_and_logged(self, mock_post):
        mock_post.side_effect = requests.Timeout("timed out")

        with self.assertLogs("src.utils.telegram_notifier", level="ERROR"):
            result = send_telegram_alert("mensagem")

        self.assertFalse(result)

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"})
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_connection_error_is_handled_and_logged(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("connection refused")

        with self.assertLogs("src.utils.telegram_notifier", level="ERROR"):
            result = send_telegram_alert("mensagem")

        self.assertFalse(result)


class TestMissingConfiguration(unittest.TestCase):
    """Fase 4: BOT_TOKEN/CHAT_ID em falta não pode causar exceção nem
    tentar contactar a rede."""

    @patch.dict("os.environ", {"TELEGRAM_CHAT_ID": "chat456"}, clear=True)
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_missing_bot_token_skips_send(self, mock_post):
        result = send_telegram_alert("mensagem")

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok123"}, clear=True)
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_missing_chat_id_skips_send(self, mock_post):
        result = send_telegram_alert("mensagem")

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.utils.telegram_notifier.post_with_retry")
    def test_missing_both_skips_send(self, mock_post):
        result = send_telegram_alert("mensagem")

        self.assertFalse(result)
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
