from src.api.client import BzzoiroClient


class OddsCollector:

    def __init__(self):

        self.client = BzzoiroClient()


    def get_event_odds(self, event_id):

        data = self.client.get(
            f"odds/?event={event_id}"
        )

        odds = {}

        for item in data.get("results", []):

            if item["market"] == "1x2":

                odds[item["outcome"]] = item["decimal_odds"]

        return odds
