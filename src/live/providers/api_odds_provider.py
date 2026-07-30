from src.live.providers.odds_provider import OddsProvider


class APIOddsProvider(OddsProvider):
    """
    Provider para Odds API.
    Placeholder.
    """

    def __init__(self, api_client=None):
        self.api_client = api_client

    def get_live_odds(self, match_id):
        raise NotImplementedError(
            "Real Odds API not connected yet."
        )
