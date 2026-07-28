from src.api.client import BzzoiroClient
from src.models.match import Match
from src.model.predictor import predict_probability
from src.collector.odds import OddsCollector


class EventCollector:

    def __init__(self):

        self.client = BzzoiroClient()
        self.odds = OddsCollector()


    def get_matches(self, limit=10):

        data = self.client.get(
            f"events/?limit={limit}"
        )

        matches = []


        for event in data["results"]:

            probability = predict_probability(
                event
            )


            market = self.odds.get_event_odds(
                event["id"]
            )


            matches.append(
                Match(
                    home=event["home_team"],
                    away=event["away_team"],
                    odds=market,
                    probability=probability,
                    league=str(
                        event["league_id"]
                    ),
                    xg_home=None,
                    xg_away=None,
                    confidence=50
                )
            )


        return matches
