from src.api.client import BzzoiroClient


client = BzzoiroClient()


endpoints = [
    "odds/?limit=1",
    "markets/?limit=1",
    "event-odds/?limit=1",
    "events/215227/odds/",
]


for endpoint in endpoints:

    print("\nTEST:", endpoint)

    try:
        data = client.get(endpoint)
        print("OK")
        print(data)

    except Exception as e:
        print("ERRO:", e)
