from src.collector.odds import OddsCollector


collector = OddsCollector()


odds = collector.get_event_odds(3277)


print(odds)
