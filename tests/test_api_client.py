import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
