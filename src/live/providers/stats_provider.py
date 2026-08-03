import os

from src.api.http_retry import get_with_retry


class StatsProvider:

    BASE_URL = "https://sports.bzzoiro.com"


    def __init__(self):

        self.api_key = os.getenv(
            "BSD_API_KEY"
        )


    def headers(self):

        return {
            "Authorization":
            f"Token {self.api_key}"
        }


    def get_event_stats(self, event_id):

        r = get_with_retry(
            f"{self.BASE_URL}/api/v2/events/{event_id}/stats/",
            headers=self.headers(),
            timeout=10
        )

        r.raise_for_status()

        return r.json()
