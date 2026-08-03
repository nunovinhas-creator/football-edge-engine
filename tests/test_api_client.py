import unittest
from unittest.mock import MagicMock, patch

from src.api.client import BzzoiroClient
from src.config.settings import MissingAPIKeyError, SUPPORTED_API_KEY_ENV_VARS


class TestBzzoiroClientMissingApiKey(unittest.TestCase):

    @patch("src.config.settings.API_KEY", None)
    def test_missing_api_key_raises_friendly_error_instead_of_attribute_error(self):
        with self.assertRaises(MissingAPIKeyError) as ctx:
            BzzoiroClient()

        message = str(ctx.exception)

        for env_var in SUPPORTED_API_KEY_ENV_VARS:
            self.assertIn(env_var, message)

        self.assertIn(".env", message)

    @patch("src.config.settings.API_KEY", "")
    def test_empty_api_key_also_raises_friendly_error(self):
        with self.assertRaises(MissingAPIKeyError):
            BzzoiroClient()

    @patch("src.config.settings.API_KEY", "some-real-key")
    def test_present_api_key_initializes_client_normally(self):
        client = BzzoiroClient()
        self.assertEqual(client.api_key, "some-real-key")


class TestBzzoiroClientAuthHeader(unittest.TestCase):
    """Garante que o header de autenticação segue a especificação OpenAPI da
    BSD Sports API (schema.yaml): `Authorization: Token <API_KEY>`."""

    @patch("src.api.client.get_with_retry")
    @patch("src.config.settings.API_KEY", "some-real-key")
    def test_get_sends_authorization_token_header(self, mock_get_with_retry):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_get_with_retry.return_value = mock_response

        client = BzzoiroClient()
        client.get("events/?limit=10")

        _, kwargs = mock_get_with_retry.call_args
        headers = kwargs["headers"]

        self.assertEqual(headers["Authorization"], "Token some-real-key")
        self.assertNotIn("X-API-Key", headers)

    @patch("src.api.client.get_with_retry")
    @patch("src.config.settings.API_KEY", "some-real-key")
    def test_get_calls_raise_for_status_and_returns_json(self, mock_get_with_retry):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": ["ok"]}
        mock_get_with_retry.return_value = mock_response

        client = BzzoiroClient()
        result = client.get("events/?limit=10")

        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(result, {"results": ["ok"]})


if __name__ == "__main__":
    unittest.main()
