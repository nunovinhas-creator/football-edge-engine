from src.api.client import BzzoiroClient


client = BzzoiroClient()

events = set()


for offset in range(0, 50000, 100):

    data = client.get(
        f"odds/?limit=100&offset={offset}"
    )

    results = data.get(
        "results",
        []
    )

    if not results:
        break

    for odd in results:
        events.add(
            odd["event_id"]
        )

    print(
        "offset:",
        offset,
        "event_ids:",
        len(events)
    )


print("----------------")
print("TOTAL EVENT IDS:")
print(len(events))

print(list(events)[:20])
