from src.live.providers.match_provider import MatchProvider


class APIMatchProvider(MatchProvider):
    """
    Provider para API real.
    Nesta fase é apenas um placeholder.
    """

    def __init__(self, api_client=None):
        self.api_client = api_client

    def get_live_match(self, match_id):
        raise NotImplementedError(
            "Real API provider not connected yet."
        )
