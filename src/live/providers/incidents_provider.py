from src.api.http_retry import get_with_retry
from src.config.settings import API_KEY, BSD_ROOT_URL


class IncidentsProvider:

    BASE_URL = BSD_ROOT_URL

    def __init__(self):
        self.api_key = API_KEY

    def get_incidents(self, event_id):

        r = get_with_retry(
            f"{self.BASE_URL}/api/v2/events/{event_id}/incidents/",
            headers={
                "Authorization": f"Token {self.api_key}"
            },
            timeout=10
        )

        r.raise_for_status()

        return r.json().get(
            "incidents",
            []
        )
