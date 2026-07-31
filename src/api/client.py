import requests
from src.config.settings import API_KEY, BASE_URL


class BzzoiroClient:

    def __init__(self):
        self.api_key = API_KEY.rstrip("/")
        self.base_url = BASE_URL.rstrip("/")

    def get(self, endpoint: str):

        endpoint = endpoint.lstrip("/")

        url = f"{self.base_url}/{endpoint}"

        response = requests.get(
            url,
            headers={
                "X-API-Key": self.api_key,
                "Accept": "application/json"
            },
            timeout=30,
        )

        response.raise_for_status()

        return response.json()
