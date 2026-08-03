from src.api.http_retry import get_with_retry
from src.config.settings import API_KEY, BSD_ROOT_URL


class StatsProvider:

    BASE_URL = BSD_ROOT_URL


    def __init__(self):

        self.api_key = API_KEY


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
