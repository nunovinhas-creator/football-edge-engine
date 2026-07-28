from src.api.client import BzzoiroClient
from src.collector.odds import OddsCollector


client = BzzoiroClient()
odds = OddsCollector()


events = client.get(
    "events/?limit=3"
)


for event in events["results"]:

    print("----------------")
    print(
        event["home_team"],
        "vs",
        event["away_team"]
    )

    print(
        odds.get_event_odds(
            event["id"]
        )
    )
