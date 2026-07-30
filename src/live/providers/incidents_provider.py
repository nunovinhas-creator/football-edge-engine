import os
import requests


class IncidentsProvider:

    BASE_URL = "https://sports.bzzoiro.com"

    def __init__(self):
        self.api_key = os.getenv("BSD_API_KEY")

    def get_incidents(self, event_id):

        r = requests.get(
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
