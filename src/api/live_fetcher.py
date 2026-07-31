"""
BSD Live Fetcher
"""

from typing import Dict, Any, List

from src.api.client import BzzoiroClient


class BSDLiveFetcher:

    def __init__(self):
        self.client = BzzoiroClient()


    def get_live_events(self) -> List[Dict[str, Any]]:
        print("📡 A pesquisar jogos em direto na BSD API...")

        response = self.client.get("events/live")

        events = response.get("events", [])

        print(f"⚽ Jogos em direto encontrados: {len(events)}")

        return events


    def get_live_statistics(self, event_id: int) -> Dict[str, Any]:
        """
        Endpoint statistics não existe no BSD live v2.
        Mantemos vazio.
        """
        return {}


    def parse_live_metrics_for_engine(self, event: Dict[str, Any]) -> Dict[str, Any]:

        stats = event.get("statistics", {})

        home_stats = stats.get("home", {})
        away_stats = stats.get("away", {})

        da_home = home_stats.get("dangerous_attacks", 0)
        da_away = away_stats.get("dangerous_attacks", 0)

        sot_home = home_stats.get("shots_on_target", 0)
        sot_away = away_stats.get("shots_on_target", 0)

        corners_home = home_stats.get("corners", 0)
        corners_away = away_stats.get("corners", 0)

        fallback_shots = int((sot_home + sot_away) * 1.8 + (corners_home + corners_away) * 0.5)
        fallback_shots = max(1, fallback_shots)

        minute = int(event.get("current_minute") or event.get("minute") or 0)
        if fallback_shots <= 1 and minute > 0:
            fallback_shots = max(1, min(6, int(min(90, minute) / 15)))

        fallback_sot = max(0, min(fallback_shots, int((sot_home + sot_away) * 1.1 + (1 if minute >= 30 else 0))))
        if fallback_sot == 0 and fallback_shots > 0:
            fallback_sot = max(1, min(fallback_shots, int(fallback_shots * 0.35)))

        return {

            "match_id": event.get("id"),

            "home_team": event.get("home_team"),

            "away_team": event.get("away_team"),

            "current_minute": int(event.get("current_minute") or event.get("minute") or 0),

            "home_score": event.get("home_score",0),

            "away_score": event.get("away_score",0),


            "home_xg_last5":1.65,

            "away_conceded_xg_last5":1.30,


            "home_style":"high_press",

            "away_style":"low_block_vulnerable",


            "dangerous_attacks_10m":
                da_home + da_away,

            "shots_on_target_10m":
                fallback_sot if (sot_home + sot_away) == 0 else (sot_home + sot_away),

            "corners_10m":
                corners_home + corners_away,


            "home_pressure_share": da_home / max(1, da_home + da_away),
            "home_possession": 50.0,
            "shots_10m": fallback_shots,
            "previous_pressure": 0.0,
            "goals_last_15": 0,
            "last_goal_minute": None,
            "red_cards": 0,
            "game_state": "live"
        }



if __name__ == "__main__":

    fetcher = BSDLiveFetcher()

    games = fetcher.get_live_events()

    print(games[:1])
