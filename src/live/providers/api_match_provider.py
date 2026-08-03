from src.api.http_retry import get_with_retry
from src.config.settings import API_KEY, BSD_ROOT_URL
from src.models.live_state import LiveMatchState
from src.live.providers.stats_provider import StatsProvider
from src.live.providers.incidents_provider import IncidentsProvider
from src.live.providers.bsd_feature_adapter import BSDFeatureAdapter


class APIMatchProvider:

    BASE_URL = BSD_ROOT_URL


    def __init__(self):

        self.api_key = API_KEY

        self.stats_provider = StatsProvider()

        self.incidents_provider = IncidentsProvider()

        self.feature_adapter = BSDFeatureAdapter()


    def headers(self):

        return {
            "Authorization": f"Token {self.api_key}"
        }


    def get_live_matches(self):

        r = get_with_retry(
            f"{self.BASE_URL}/api/v2/events/live/",
            headers=self.headers(),
            timeout=10
        )

        r.raise_for_status()

        return r.json()


    def get_live_match(self, match_id):

        event = get_with_retry(
            f"{self.BASE_URL}/api/v2/events/{match_id}/?full=true",
            headers=self.headers(),
            timeout=10
        ).json()


        stats = self.stats_provider.get_event_stats(
            match_id
        )


        match_stats = stats.get(
            "stats",
            {}
        )


        home = match_stats.get(
            "home",
            {}
        )


        away = match_stats.get(
            "away",
            {}
        )


        home_xg = (
            home.get("xg", {})
            .get("actual")
            or 1.0
        )


        away_xg = (
            away.get("xg", {})
            .get("actual")
            or 1.0
        )


        incidents = self.incidents_provider.get_incidents(
            match_id
        )


        incident_features = self.feature_adapter.incidents_to_features(
            incidents,
            event.get("current_minute", 0)
        )


        return LiveMatchState(

            minute=event.get(
                "current_minute",
                0
            ),

            home_score=event.get(
                "home_score",
                0
            ),

            away_score=event.get(
                "away_score",
                0
            ),


            home_xg_last5=home_xg,

            away_conceded_xg_last5=away_xg,


            home_style="balanced",


            dangerous_attacks_10m=0,

            shots_on_target_10m=0,

            shots_10m=0,

            corners_10m=0,


            possession=50.0,


            goals_last_15=incident_features[
                "goals_last_15"
            ],

            last_goal_minute=incident_features[
                "last_goal_minute"
            ],

            red_cards=incident_features[
                "red_cards"
            ],

            game_state=incident_features[
                "game_state"
            ]
        )
