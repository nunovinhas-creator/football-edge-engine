"""
Testes unitários do cliente HTTP dedicado (src/historical_dataset/client.py).

Mocka `get_with_retry` (o mesmo helper de retries usado por todo o
projeto) para verificar: cabeçalho de autenticação correto
(`Authorization: Token <key>`, esquema `tokenAuth` de schema.yaml), rate
limiting aplicado antes de cada pedido, tratamento de respostas 4xx e de
corpo vazio.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.historical_dataset.client import BSDAPIError, BSDHistoricalClient
from src.historical_dataset.rate_limiter import RateLimiter


def _fake_response(status_code=200, json_body=None, content=b"{}", text=""):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.content = content
    response.text = text
    return response


class TestBSDHistoricalClient(unittest.TestCase):

    def test_sends_authorization_token_header(self):
        client = BSDHistoricalClient(api_key="secret123", base_url="https://example.test/api/v2")

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(json_body=[{"id": 1}])
            client.get("events/", params={"limit": 10})

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Token secret123")
        self.assertEqual(kwargs["headers"]["Accept"], "application/json")

    def test_builds_url_from_base_url_and_endpoint(self):
        client = BSDHistoricalClient(api_key="k", base_url="https://example.test/api/v2")

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(json_body=[])
            client.get("/leagues/")

        called_url = mock_get.call_args[0][0]
        self.assertEqual(called_url, "https://example.test/api/v2/leagues/")

    def test_returns_parsed_json(self):
        client = BSDHistoricalClient(api_key="k")
        payload = [{"id": 1, "name": "Premier League"}]

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(json_body=payload)
            result = client.get("leagues/")

        self.assertEqual(result, payload)

    def test_empty_body_returns_none(self):
        client = BSDHistoricalClient(api_key="k")

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(status_code=204, content=b"")
            result = client.get("events/1/odds/")

        self.assertIsNone(result)

    def test_4xx_raises_bsd_api_error(self):
        client = BSDHistoricalClient(api_key="k")

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(status_code=404, text="not found")
            with self.assertRaises(BSDAPIError) as ctx:
                client.get("events/999/")

        self.assertEqual(ctx.exception.status_code, 404)

    def test_rate_limiter_acquire_called_before_request(self):
        rate_limiter = MagicMock(spec=RateLimiter)
        client = BSDHistoricalClient(api_key="k", rate_limiter=rate_limiter)

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(json_body=[])
            client.get("leagues/")

        rate_limiter.acquire.assert_called_once()

    def test_request_count_starts_at_zero(self):
        client = BSDHistoricalClient(api_key="k")
        self.assertEqual(client.request_count, 0)

    def test_request_count_increments_on_each_successful_request(self):
        client = BSDHistoricalClient(api_key="k")

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(json_body=[])
            client.get("leagues/")
            client.get("events/")

        self.assertEqual(client.request_count, 2)

    def test_request_count_increments_even_on_4xx(self):
        """A contagem reflete pedidos feitos, não apenas os bem-sucedidos (para dataset_report.json)."""
        client = BSDHistoricalClient(api_key="k")

        with patch("src.historical_dataset.client.get_with_retry") as mock_get:
            mock_get.return_value = _fake_response(status_code=404, text="not found")
            with self.assertRaises(BSDAPIError):
                client.get("events/999/")

        self.assertEqual(client.request_count, 1)


if __name__ == "__main__":
    unittest.main()
