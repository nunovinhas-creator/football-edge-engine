from src.live.providers.match_provider import MatchProvider
from src.models.live_state import LiveMatchState


class MockMatchProvider(MatchProvider):

    def get_live_match(self, match_id):

        return LiveMatchState(
            minute=75,
            possession=60.0,
            dangerous_attacks_10m=12,
            shots_on_target_10m=4,
            shots_10m=8,
            corners_10m=3,
            previous_pressure=55.0
        )
