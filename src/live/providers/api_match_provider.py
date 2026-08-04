from src.api.http_retry import get_with_retry
from src.config.settings import API_KEY, BSD_ROOT_URL
from src.models.live_state import LiveMatchState
from src.live.providers.stats_provider import StatsProvider
from src.live.providers.incidents_provider import IncidentsProvider
from src.live.providers.bsd_feature_adapter import BSDFeatureAdapter


def _lookup_stat(container, *keys):
    """Procura o primeiro de `keys` num dict de stats por equipa (case-insensitive).

    Mesmo padrão defensivo usado em `historical_dataset/normalizer.py` para
    o mesmo endpoint (`/events/{id}/stats/`), já que a BSD API não
    documenta formalmente esta resposta (schema.yaml marca-a como
    "No response body").
    """

    if not isinstance(container, dict):
        return None

    for key in keys:
        if key in container and container[key] is not None:
            return container[key]

    lowered = {str(k).lower(): v for k, v in container.items()}

    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value

    return None


def _to_number(value, default=0.0):
    if value is None:
        return default

    if isinstance(value, str):
        value = value.replace("%", "").strip()
        if not value:
            return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


        dangerous_attacks = int(
            _to_number(_lookup_stat(home, "dangerous_attack", "dangerous_attacks"))
            + _to_number(_lookup_stat(away, "dangerous_attack", "dangerous_attacks"))
        )

        shots = int(
            _to_number(_lookup_stat(home, "shots_total", "total_shots", "shots"))
            + _to_number(_lookup_stat(away, "shots_total", "total_shots", "shots"))
        )

        shots_on_target = int(
            _to_number(_lookup_stat(home, "shots_on_target", "shots_on_goal", "on_target_shots"))
            + _to_number(_lookup_stat(away, "shots_on_target", "shots_on_goal", "on_target_shots"))
        )

        corners = int(
            _to_number(_lookup_stat(home, "corners", "corner_kicks"))
            + _to_number(_lookup_stat(away, "corners", "corner_kicks"))
        )

        possession = _to_number(
            _lookup_stat(home, "possession", "ball_possession", "possession_pct"),
            default=None
        )

        if possession is None:
            away_possession = _to_number(
                _lookup_stat(away, "possession", "ball_possession", "possession_pct"),
                default=None
            )

            possession = (
                (100.0 - away_possession)
                if away_possession is not None
                else 50.0
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


            dangerous_attacks_10m=dangerous_attacks,

            shots_on_target_10m=shots_on_target,

            shots_10m=shots,

            corners_10m=corners,


            possession=possession,


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
