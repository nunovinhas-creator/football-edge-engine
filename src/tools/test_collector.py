from src.collector.client import EventCollector


collector = EventCollector()


matches = collector.get_matches(3)


for match in matches:

    print("----------------")
    print(match.to_dict())
