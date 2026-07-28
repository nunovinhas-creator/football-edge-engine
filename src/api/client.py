import requests

from src.config.settings import API_KEY, BASE_URL


class BzzoiroClient:

    def __init__(self):
        self.headers = {
            "Authorization": f"Token {API_KEY}"
        }

    def get(self, endpoint: str):

        url = f"{BASE_URL}/{endpoint}"

        response = requests.get(
            url,
            headers=self.headers,
            timeout=30
        )

        response.raise_for_status()

        return response.json()
