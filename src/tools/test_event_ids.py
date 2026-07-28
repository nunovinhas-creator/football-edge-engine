from src.api.client import BzzoiroClient


client = BzzoiroClient()


data = client.get(
    "events/?limit=3"
)


for event in data["results"]:

    print("----------------")
    print("ID:", event["id"])
    print(
        event["home_team"],
        "vs",
        event["away_team"]
    )
