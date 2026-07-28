from src.collector.client import EventCollector
from src.engine.full_engine import analyze_match


collector = EventCollector()


matches = collector.get_matches(3)


for match in matches:

    print("====================")

    results = analyze_match(
        match
    )

    for result in results:

        print(result)
