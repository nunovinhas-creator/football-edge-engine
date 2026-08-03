from src.api.http_retry import get_with_retry
from src.config.settings import BASE_URL, require_api_key


class BzzoiroClient:

    def __init__(self):
        self.api_key = require_api_key().rstrip("/")
        self.base_url = BASE_URL.rstrip("/")

    def get(self, endpoint: str):

        endpoint = endpoint.lstrip("/")

        url = f"{self.base_url}/{endpoint}"

        response = get_with_retry(
            url,
            headers={
                "X-API-Key": self.api_key,
                "Accept": "application/json"
            },
            timeout=30,
        )

        response.raise_for_status()

        return response.json()
