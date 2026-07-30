from src.live.providers.odds_provider import OddsProvider


class MockOddsProvider(OddsProvider):

    def get_live_odds(self, match_id):

        return {
            "next_goal": 2.10,
            "over_1_5": 2.15,
            "over_2_5": 3.40,
            "btts": 2.80
        }
