import os

from src.api.http_retry import get_with_retry


class APIOddsProvider:

    BASE_URL = "https://sports.bzzoiro.com"


    def __init__(self):

        self.api_key = os.getenv(
            "BSD_API_KEY"
        )

        if not self.api_key:
            raise RuntimeError(
                "BSD_API_KEY missing"
            )


    def headers(self):

        return {
            "Authorization":
            f"Token {self.api_key}"
        }


    def get_live_odds(self, event_id):

        url = (
            f"{self.BASE_URL}"
            f"/api/v2/events/{event_id}/odds/"
        )


        response = get_with_retry(
            url,
            headers=self.headers(),
            timeout=10
        )


        response.raise_for_status()

        return response.json()
